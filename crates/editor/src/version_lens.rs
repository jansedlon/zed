//! "Version lens": above each outdated dependency, shows clickable code-lens
//! badges for the newest **patch**, **minor**, and **major** release available;
//! clicking one rewrites the dependency to that version (preserving the `^`/`~`
//! prefix). Works in `package.json` (deps/devDeps/peer/optional) and in
//! `pnpm-workspace.yaml` (`catalog:` + named `catalogs:`), where pnpm versions
//! are centrally managed.
//!
//! In `package.json`, pnpm `catalog:` / `catalog:<name>` and `workspace:` specs
//! are resolved against the workspace's `pnpm-workspace.yaml` and sibling
//! `package.json` files (read once per worktree, cached) and shown as a muted,
//! read-only hint of what they resolve to — the actual version lives in the
//! catalog (clickable there) or the local package.
//!
//! The full version list is fetched from the npm registry once per session,
//! cached globally, and fetched in the background so rendering never blocks.
//! Blocks are only shown for dependencies with a newer version, and update in
//! place (no flicker) as fetches land.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use collections::{HashMap, HashSet};
use fs::Fs;
use futures::{AsyncReadExt as _, StreamExt as _, stream};
use gpui::{App, AsyncWindowContext, Context, Global, MouseButton, Task, WeakEntity};
use http_client::{AsyncBody, HttpClient, Request};
use multi_buffer::{Anchor, MultiBufferSnapshot, ToPoint as _};
use project::WorktreeId;
use semver::Version;
use text::Point;
use ui::prelude::*;
use util::ResultExt as _;

use crate::{
    Editor,
    display_map::{
        BlockContext, BlockPlacement, BlockProperties, BlockStyle, CustomBlockId, RenderBlock,
    },
};

const REGISTRY: &str = "https://registry.npmjs.org";
const FETCH_CONCURRENCY: usize = 8;
const DEBOUNCE: Duration = Duration::from_millis(150);
/// Cap on README markdown kept for the hover popover, to keep parsing snappy.
const README_CAP: usize = 24_000;
const PACKAGE_JSON: &str = "package.json";
const PNPM_WORKSPACE: &str = "pnpm-workspace.yaml";
const JSON_SECTIONS: [&str; 4] = [
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
];

#[derive(Clone, Copy)]
enum LensFile {
    PackageJson,
    PnpmWorkspace,
}

fn file_kind(name: &str) -> Option<LensFile> {
    match name {
        PACKAGE_JSON => Some(LensFile::PackageJson),
        PNPM_WORKSPACE => Some(LensFile::PnpmWorkspace),
        _ => None,
    }
}

#[derive(Clone)]
enum PackageState {
    Fetching,
    /// All stable (non-prerelease) versions, sorted ascending. Empty on failure.
    Resolved(Arc<[Version]>),
}

#[derive(Default)]
struct VersionCacheInner {
    packages: HashMap<String, PackageState>,
}

#[derive(Clone, Default)]
struct VersionCache(Arc<Mutex<VersionCacheInner>>);
impl Global for VersionCache {}

fn version_cache(cx: &mut App) -> VersionCache {
    cx.default_global::<VersionCache>().clone()
}

/// Resolved pnpm workspace metadata for one worktree, built lazily in the
/// background and cached. Used to resolve `catalog:`/`workspace:` specs in
/// `package.json`.
#[derive(Default)]
struct WorkspaceContext {
    /// Default `catalog:` map (package -> version spec).
    default_catalog: HashMap<String, String>,
    /// Named `catalogs:` maps (catalog name -> package -> version spec).
    named_catalogs: HashMap<String, HashMap<String, String>>,
    /// Local package name -> version, from every `package.json` in the worktree.
    local_versions: HashMap<String, String>,
}

enum WorkspaceState {
    Building,
    Ready(Arc<WorkspaceContext>),
}

#[derive(Default)]
struct WorkspaceCacheInner {
    by_worktree: HashMap<WorktreeId, WorkspaceState>,
}

#[derive(Clone, Default)]
struct WorkspaceCache(Arc<Mutex<WorkspaceCacheInner>>);
impl Global for WorkspaceCache {}

fn workspace_cache(cx: &mut App) -> WorkspaceCache {
    cx.default_global::<WorkspaceCache>().clone()
}

/// npm registry metadata for one package, used to render the hover popover.
struct PackageDoc {
    latest: Option<Version>,
    description: Option<String>,
    homepage: Option<String>,
    repository: Option<String>,
    license: Option<String>,
    readme: Option<String>,
}

#[derive(Clone)]
enum DocState {
    /// Fetched but unavailable (network error / 404). Cached so we don't retry.
    Failed,
    Ready(Arc<PackageDoc>),
}

#[derive(Default)]
struct DocCacheInner {
    docs: HashMap<String, DocState>,
}

#[derive(Clone, Default)]
struct DocCache(Arc<Mutex<DocCacheInner>>);
impl Global for DocCache {}

fn doc_cache(cx: &mut App) -> DocCache {
    cx.default_global::<DocCache>().clone()
}

/// What a lens block displays above a dependency line. Every checkable
/// dependency always reserves exactly one of these (starting as `Loading`), so
/// the layout never shifts as versions resolve in place.
#[derive(Clone)]
enum Lens {
    /// Placeholder shown while registry / workspace data is still loading.
    Loading,
    /// Clickable patch/minor/major chips that rewrite the version on click.
    Tiers(Vec<Badge>),
    /// A muted, read-only label: "✓ latest", a resolved `catalog:`/`workspace:`
    /// hint, or an unavailable/unresolved note.
    Note(SharedString),
}

fn lens_signature(lens: &Lens) -> String {
    match lens {
        Lens::Loading => "loading".to_string(),
        Lens::Tiers(badges) => badge_signature(badges),
        Lens::Note(text) => format!("note:{text}"),
    }
}

/// A dependency entry. `context` is the section it lives in — a `package.json`
/// section (`"dependencies"`, …) or a catalog (`"catalog"` / `"catalogs.<name>"`)
/// — which, with `name`, uniquely identifies it (the same package can appear in
/// multiple named catalogs).
struct Dependency {
    name: String,
    spec: String,
    context: String,
    row: u32,
    indent: u32,
    /// Column range of the package name (key) within `row`, for hover detection.
    name_cols: Range<u32>,
    /// Column range of the version value within `row`, for click-to-update.
    value_cols: Range<u32>,
}

#[derive(Default)]
struct Tiers {
    patch: Option<Version>,
    minor: Option<Version>,
    major: Option<Version>,
}

impl Tiers {
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.patch.is_none() && self.minor.is_none() && self.major.is_none()
    }

    fn badges(&self) -> Vec<Badge> {
        let mut badges = Vec::new();
        if let Some(version) = &self.patch {
            badges.push(Badge::new("patch", version.clone(), Color::Success));
        }
        if let Some(version) = &self.minor {
            badges.push(Badge::new("minor", version.clone(), Color::Warning));
        }
        if let Some(version) = &self.major {
            badges.push(Badge::new("major", version.clone(), Color::Error));
        }
        badges
    }
}

#[derive(Clone)]
struct Badge {
    label: SharedString,
    version: Version,
    color: Color,
}

impl Badge {
    fn new(tier: &str, version: Version, color: Color) -> Self {
        Self {
            label: format!("↑ {tier} {version}").into(),
            version,
            color,
        }
    }
}

struct LensBlock {
    id: CustomBlockId,
    signature: String,
}

pub(super) struct VersionLensState {
    /// Keyed by `(context, name)` so the same package in different catalogs gets
    /// its own block. Each block's anchor tracks its line automatically.
    blocks: HashMap<(String, String), LensBlock>,
    refresh_task: Task<()>,
    /// Whether the initial (immediate) loading-slot placement has happened, so
    /// later refreshes can debounce without delaying the first paint.
    initialized: bool,
}

impl VersionLensState {
    fn new() -> Self {
        Self {
            blocks: HashMap::default(),
            refresh_task: Task::ready(()),
            initialized: false,
        }
    }
}

impl Editor {
    fn version_lens_kind(&self, cx: &App) -> Option<LensFile> {
        let buffer = self.buffer().read(cx).as_singleton()?;
        let buffer = buffer.read(cx);
        file_kind(buffer.file()?.file_name(cx))
    }

    fn version_lens_worktree_id(&self, cx: &App) -> Option<WorktreeId> {
        let buffer = self.buffer().read(cx).as_singleton()?;
        Some(buffer.read(cx).file()?.worktree_id(cx))
    }

    /// Returns the cached pnpm workspace context for the current worktree,
    /// spawning a background build (and re-render) the first time it is needed.
    fn version_lens_workspace_context(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Option<Arc<WorkspaceContext>> {
        let project = self.project.clone()?;
        let worktree_id = self.version_lens_worktree_id(cx)?;

        let cache = workspace_cache(cx);
        {
            let mut guard = cache.0.lock().unwrap();
            match guard.by_worktree.get(&worktree_id) {
                Some(WorkspaceState::Ready(context)) => return Some(context.clone()),
                Some(WorkspaceState::Building) => return None,
                None => {
                    guard
                        .by_worktree
                        .insert(worktree_id, WorkspaceState::Building);
                }
            }
        }

        // Gather inputs on the main thread, then read files in the background.
        let project = project.read(cx);
        let fs = project.fs().clone();
        let Some(worktree) = project.worktree_for_id(worktree_id, cx) else {
            cache.0.lock().unwrap().by_worktree.remove(&worktree_id);
            return None;
        };
        let worktree = worktree.read(cx);
        let workspace_yaml = worktree.abs_path().join(PNPM_WORKSPACE);
        let package_jsons: Vec<PathBuf> = worktree
            .files(false, 0)
            .filter(|entry| {
                entry.path.file_name() == Some(PACKAGE_JSON)
                    && !entry.path.as_unix_str().contains("node_modules")
            })
            .map(|entry| worktree.absolutize(&entry.path))
            .collect();

        cx.spawn(async move |editor, cx| {
            let context = build_workspace_context(fs, workspace_yaml, package_jsons).await;
            cache
                .0
                .lock()
                .unwrap()
                .by_worktree
                .insert(worktree_id, WorkspaceState::Ready(Arc::new(context)));
            editor
                .update(cx, |editor, cx| editor.update_version_lens(cx))
                .ok();
        })
        .detach();
        None
    }

    /// Entry point, called whenever the buffer is (re)parsed. The very first
    /// pass runs immediately so loading slots are placed before the user starts
    /// interacting (no jump); later passes debounce to coalesce edits.
    pub(crate) fn schedule_version_lens_refresh(&mut self, cx: &mut Context<Self>) {
        if self.version_lens_kind(cx).is_none() {
            self.clear_version_lens(cx);
            return;
        }
        let initialized = self
            .version_lens
            .as_ref()
            .is_some_and(|state| state.initialized);
        if !initialized {
            self.version_lens
                .get_or_insert_with(VersionLensState::new)
                .initialized = true;
            self.update_version_lens(cx);
            return;
        }
        let task = cx.spawn(async move |editor, cx| {
            cx.background_executor().timer(DEBOUNCE).await;
            editor
                .update(cx, |editor, cx| editor.update_version_lens(cx))
                .ok();
        });
        self.version_lens
            .get_or_insert_with(VersionLensState::new)
            .refresh_task = task;
    }

    fn clear_version_lens(&mut self, cx: &mut Context<Self>) {
        let Some(state) = self.version_lens.as_mut() else {
            return;
        };
        let ids: HashSet<CustomBlockId> = state.blocks.drain().map(|(_, block)| block.id).collect();
        if !ids.is_empty() {
            self.remove_blocks(ids, None, cx);
        }
    }

    fn update_version_lens(&mut self, cx: &mut Context<Self>) {
        let Some(kind) = self.version_lens_kind(cx) else {
            self.clear_version_lens(cx);
            return;
        };

        let snapshot = self.buffer().read(cx).snapshot(cx);
        let deps = parse_deps(&snapshot.text(), kind);

        // Editing the catalog file changes what `catalog:` specs resolve to, so
        // drop the cached workspace context for this worktree.
        if matches!(kind, LensFile::PnpmWorkspace) {
            if let Some(worktree_id) = self.version_lens_worktree_id(cx) {
                workspace_cache(cx)
                    .0
                    .lock()
                    .unwrap()
                    .by_worktree
                    .remove(&worktree_id);
            }
        }

        // Only `package.json` references catalogs/workspace packages; resolving
        // them needs the (lazily built) workspace context.
        let workspace = if matches!(kind, LensFile::PackageJson)
            && deps.iter().any(|dep| needs_workspace_context(&dep.spec))
        {
            self.version_lens_workspace_context(cx)
        } else {
            None
        };

        let cache = version_cache(cx);
        let mut to_fetch = Vec::new();
        // (context, name, row, indent, lens) for every checkable dependency. A
        // lens is always reserved (`Loading` until resolved) so the layout is
        // stable from the first paint and never shifts as data lands.
        let mut desired: Vec<(String, String, u32, u32, Lens)> = Vec::new();
        {
            let mut guard = cache.0.lock().unwrap();
            for dep in &deps {
                let lens = if let Some(current) = spec_version(&dep.spec) {
                    match guard.packages.get(&dep.name) {
                        Some(PackageState::Resolved(versions)) => {
                            if versions.is_empty() {
                                Lens::Note("× unavailable".into())
                            } else {
                                let badges = compute_tiers(&current, versions).badges();
                                if badges.is_empty() {
                                    Lens::Note("✓ latest".into())
                                } else {
                                    Lens::Tiers(badges)
                                }
                            }
                        }
                        Some(PackageState::Fetching) => Lens::Loading,
                        None => {
                            guard
                                .packages
                                .insert(dep.name.clone(), PackageState::Fetching);
                            to_fetch.push(dep.name.clone());
                            Lens::Loading
                        }
                    }
                } else if needs_workspace_context(&dep.spec) {
                    match workspace.as_deref() {
                        Some(context) => match resolve_ref(&dep.spec, &dep.name, Some(context)) {
                            Some(info) => Lens::Note(info),
                            None => Lens::Note(format!("{} (unresolved)", dep.spec.trim()).into()),
                        },
                        None => Lens::Loading,
                    }
                } else {
                    // Non-checkable spec (git/file/url/…): no lens, ever — decided
                    // synchronously, so it never causes a shift.
                    continue;
                };
                desired.push((
                    dep.context.clone(),
                    dep.name.clone(),
                    dep.row,
                    dep.indent,
                    lens,
                ));
            }
        }

        self.reconcile_version_lens_blocks(desired, &snapshot, cx);

        if !to_fetch.is_empty() {
            let http = cx.http_client();
            cx.spawn(async move |editor, cx| {
                let results = stream::iter(to_fetch.into_iter().map(|package| {
                    let http = http.clone();
                    async move {
                        let versions = fetch_versions(http, &package).await;
                        (package, versions)
                    }
                }))
                .buffer_unordered(FETCH_CONCURRENCY)
                .collect::<Vec<_>>()
                .await;
                {
                    let mut guard = cache.0.lock().unwrap();
                    for (package, versions) in results {
                        guard
                            .packages
                            .insert(package, PackageState::Resolved(versions.into()));
                    }
                }
                editor
                    .update(cx, |editor, cx| editor.update_version_lens(cx))
                    .ok();
            })
            .detach();
        }
    }

    fn reconcile_version_lens_blocks(
        &mut self,
        desired: Vec<(String, String, u32, u32, Lens)>,
        snapshot: &MultiBufferSnapshot,
        cx: &mut Context<Self>,
    ) {
        let mut existing =
            std::mem::take(&mut self.version_lens.get_or_insert_with(VersionLensState::new).blocks);

        let editor = cx.weak_entity();
        let mut keep: HashMap<(String, String), LensBlock> = HashMap::default();
        let mut renderers_to_replace: HashMap<CustomBlockId, RenderBlock> = HashMap::default();
        let mut to_insert: Vec<((String, String), BlockProperties<Anchor>, String)> = Vec::new();

        for (context, name, row, indent, lens) in desired {
            let key = (context.clone(), name.clone());
            let signature = lens_signature(&lens);
            let renderer = || {
                build_renderer(
                    editor.clone(),
                    context.clone(),
                    name.clone().into(),
                    lens.clone(),
                    indent,
                )
            };
            if let Some(block) = existing.remove(&key) {
                if block.signature != signature {
                    renderers_to_replace.insert(block.id, renderer());
                }
                keep.insert(key, LensBlock { id: block.id, signature });
            } else {
                let anchor = snapshot.anchor_before(Point::new(row, 0));
                let props = BlockProperties {
                    placement: BlockPlacement::Above(anchor),
                    height: Some(1),
                    style: BlockStyle::Spacer,
                    render: renderer(),
                    priority: 0,
                };
                to_insert.push((key, props, signature));
            }
        }

        let to_remove: HashSet<CustomBlockId> =
            existing.into_values().map(|block| block.id).collect();
        if !to_remove.is_empty() {
            self.remove_blocks(to_remove, None, cx);
        }
        if !renderers_to_replace.is_empty() {
            self.replace_blocks(renderers_to_replace, None, cx);
        }
        if !to_insert.is_empty() {
            let mut props = Vec::with_capacity(to_insert.len());
            let mut metadata = Vec::with_capacity(to_insert.len());
            for (key, block_props, signature) in to_insert {
                props.push(block_props);
                metadata.push((key, signature));
            }
            let ids = self.insert_blocks(props, None, cx);
            for (id, (key, signature)) in ids.into_iter().zip(metadata) {
                keep.insert(key, LensBlock { id, signature });
            }
        }

        self.version_lens
            .get_or_insert_with(VersionLensState::new)
            .blocks = keep;
    }
}

fn badge_signature(badges: &[Badge]) -> String {
    badges
        .iter()
        .map(|badge| format!("{}:{}", badge.label, badge.version))
        .collect::<Vec<_>>()
        .join("|")
}

fn build_renderer(
    editor: WeakEntity<Editor>,
    context: String,
    name: SharedString,
    lens: Lens,
    indent: u32,
) -> RenderBlock {
    match lens {
        Lens::Tiers(badges) => build_tiers_renderer(editor, context, name, badges, indent),
        Lens::Loading => build_note_renderer("checking…".into(), indent),
        Lens::Note(text) => build_note_renderer(text, indent),
    }
}

fn build_tiers_renderer(
    editor: WeakEntity<Editor>,
    context: String,
    name: SharedString,
    badges: Vec<Badge>,
    indent: u32,
) -> RenderBlock {
    let context: SharedString = context.into();
    Arc::new(move |cx: &mut BlockContext| {
        let hover_bg = cx.app.theme().colors().element_hover;
        let mut row = h_flex()
            .pl(cx.em_width * indent as f32)
            .h_full()
            .items_end()
            .gap_1();
        for (ix, badge) in badges.iter().enumerate() {
            row = row.child(
                div()
                    .id(SharedString::from(format!("version-lens-{context}-{name}-{ix}")))
                    .px_1p5()
                    .rounded_sm()
                    .cursor_pointer()
                    .hover(move |style| style.bg(hover_bg))
                    .child(
                        Label::new(badge.label.clone())
                            .size(LabelSize::Small)
                            .color(badge.color),
                    )
                    .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    .on_click({
                        let editor = editor.clone();
                        let context = context.clone();
                        let name = name.clone();
                        let version = badge.version.clone();
                        move |_event, _window, cx| {
                            if let Some(editor) = editor.upgrade() {
                                editor.update(cx, |editor, cx| {
                                    apply_version_update(editor, &context, &name, &version, cx);
                                });
                            }
                        }
                    }),
            );
        }
        row.into_any_element()
    })
}

fn build_note_renderer(text: SharedString, indent: u32) -> RenderBlock {
    Arc::new(move |cx: &mut BlockContext| {
        h_flex()
            .pl(cx.em_width * indent as f32)
            .h_full()
            .items_end()
            .child(
                Label::new(text.clone())
                    .size(LabelSize::Small)
                    .color(Color::Muted),
            )
            .into_any_element()
    })
}

/// Rewrites the dependency identified by `(context, name)` to `version`,
/// preserving its range prefix (`^`, `~`, …). Re-parses the live buffer so the
/// edit range is correct across edits.
fn apply_version_update(
    editor: &mut Editor,
    context: &str,
    name: &str,
    version: &Version,
    cx: &mut Context<Editor>,
) {
    let Some(kind) = editor.version_lens_kind(cx) else {
        return;
    };
    let snapshot = editor.buffer().read(cx).snapshot(cx);
    let Some(dep) = parse_deps(&snapshot.text(), kind)
        .into_iter()
        .find(|dep| dep.context == context && dep.name == name)
    else {
        return;
    };
    let prefix: String = dep
        .spec
        .chars()
        .take_while(|c| !c.is_ascii_digit())
        .collect();
    let new_spec = format!("{prefix}{version}");
    if new_spec == dep.spec {
        return;
    }
    let anchor_range = snapshot.anchor_before(Point::new(dep.row, dep.value_cols.start))
        ..snapshot.anchor_after(Point::new(dep.row, dep.value_cols.end));
    editor.edit([(anchor_range, new_spec)], cx);
}

fn compute_tiers(current: &Version, versions: &[Version]) -> Tiers {
    let mut tiers = Tiers::default();
    for version in versions {
        if version <= current {
            continue;
        }
        if version.major == current.major && version.minor == current.minor {
            tiers.patch = Some(version.clone());
        } else if version.major == current.major {
            tiers.minor = Some(version.clone());
        } else {
            tiers.major = Some(version.clone());
        }
    }
    tiers
}

fn parse_deps(text: &str, kind: LensFile) -> Vec<Dependency> {
    match kind {
        LensFile::PackageJson => parse_package_json(text),
        LensFile::PnpmWorkspace => parse_pnpm_catalogs(text),
    }
}

fn parse_package_json(text: &str) -> Vec<Dependency> {
    let mut deps = Vec::new();
    let mut section: Option<&'static str> = None;
    let mut section_depth = 0i32;
    let mut depth = 0i32;

    for (row, line) in text.lines().enumerate() {
        let indent = (line.len() - line.trim_start().len()) as u32;
        if let Some(context) = section {
            if let Some((name, name_cols, spec, value_cols)) = parse_json_entry(line) {
                deps.push(Dependency {
                    name,
                    spec,
                    context: context.to_string(),
                    row: row as u32,
                    indent,
                    name_cols,
                    value_cols,
                });
            }
        } else if let Some(key) = json_object_key(line.trim_start()) {
            if line.contains('{') {
                section = JSON_SECTIONS.iter().copied().find(|s| *s == key);
                if section.is_some() {
                    section_depth = depth;
                }
            }
        }

        depth += line.matches('{').count() as i32;
        depth -= line.matches('}').count() as i32;
        if section.is_some() && depth <= section_depth {
            section = None;
        }
    }
    deps
}

fn parse_pnpm_catalogs(text: &str) -> Vec<Dependency> {
    #[derive(PartialEq)]
    enum Mode {
        None,
        DefaultCatalog,
        NamedCatalogs,
    }
    let mut deps = Vec::new();
    let mut mode = Mode::None;
    let mut current_catalog: Option<String> = None;

    for (row, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = (line.len() - trimmed.len()) as u32;

        if indent == 0 {
            current_catalog = None;
            mode = match yaml_key(trimmed) {
                Some("catalog") => Mode::DefaultCatalog,
                Some("catalogs") => Mode::NamedCatalogs,
                _ => Mode::None,
            };
            continue;
        }

        match mode {
            Mode::DefaultCatalog => {
                if let Some((name, name_cols, spec, value_cols)) = parse_yaml_entry(line) {
                    deps.push(Dependency {
                        name,
                        spec,
                        context: "catalog".to_string(),
                        row: row as u32,
                        indent,
                        name_cols,
                        value_cols,
                    });
                }
            }
            Mode::NamedCatalogs => match parse_yaml_entry(line) {
                Some((name, name_cols, spec, value_cols)) => {
                    if let Some(catalog) = &current_catalog {
                        deps.push(Dependency {
                            name,
                            spec,
                            context: format!("catalogs.{catalog}"),
                            row: row as u32,
                            indent,
                            name_cols,
                            value_cols,
                        });
                    }
                }
                None => {
                    if let Some(key) = yaml_key(trimmed) {
                        current_catalog = Some(key.to_string());
                    }
                }
            },
            Mode::None => {}
        }
    }
    deps
}

/// Whether a spec must be resolved against the workspace context rather than
/// the npm registry.
fn needs_workspace_context(spec: &str) -> bool {
    let spec = spec.trim();
    spec.starts_with("catalog:") || spec.starts_with("workspace:")
}

/// Resolves a pnpm `catalog:`/`workspace:` spec to a read-only hint of what it
/// points at, using the workspace context. Returns `None` if the context is not
/// ready or the reference can't be resolved.
fn resolve_ref(spec: &str, name: &str, context: Option<&WorkspaceContext>) -> Option<SharedString> {
    let spec = spec.trim();
    let context = context?;
    if let Some(catalog) = spec.strip_prefix("catalog:") {
        let resolved = if catalog.is_empty() || catalog == "default" {
            context.default_catalog.get(name)
        } else {
            context.named_catalogs.get(catalog).and_then(|c| c.get(name))
        }?;
        Some(format!("{spec} → {resolved}").into())
    } else if spec.starts_with("workspace:") {
        let version = context.local_versions.get(name)?;
        Some(format!("{spec} → {version}").into())
    } else {
        None
    }
}

/// Reads `pnpm-workspace.yaml` and every workspace `package.json` to build the
/// resolution context. Runs on a background executor (no `cx`).
async fn build_workspace_context(
    fs: Arc<dyn Fs>,
    workspace_yaml: PathBuf,
    package_jsons: Vec<PathBuf>,
) -> WorkspaceContext {
    let mut context = WorkspaceContext::default();
    if let Ok(yaml) = fs.load(&workspace_yaml).await {
        let (default_catalog, named_catalogs) = parse_workspace_catalogs(&yaml);
        context.default_catalog = default_catalog;
        context.named_catalogs = named_catalogs;
    }
    let contents = stream::iter(package_jsons.into_iter().map(|path| {
        let fs = fs.clone();
        async move { fs.load(&path).await.ok() }
    }))
    .buffer_unordered(FETCH_CONCURRENCY)
    .collect::<Vec<_>>()
    .await;
    for text in contents.into_iter().flatten() {
        if let Some((name, version)) = parse_package_name_version(&text) {
            context.local_versions.insert(name, version);
        }
    }
    context
}

/// Splits a `pnpm-workspace.yaml` into its default `catalog:` map and named
/// `catalogs:` maps (catalog name -> package -> spec).
fn parse_workspace_catalogs(
    yaml: &str,
) -> (
    HashMap<String, String>,
    HashMap<String, HashMap<String, String>>,
) {
    let mut default_catalog = HashMap::default();
    let mut named_catalogs: HashMap<String, HashMap<String, String>> = HashMap::default();
    for dep in parse_pnpm_catalogs(yaml) {
        if dep.context == "catalog" {
            default_catalog.insert(dep.name, dep.spec);
        } else if let Some(name) = dep.context.strip_prefix("catalogs.") {
            named_catalogs
                .entry(name.to_string())
                .or_default()
                .insert(dep.name, dep.spec);
        }
    }
    (default_catalog, named_catalogs)
}

/// Extracts the top-level `name` and `version` from a `package.json`.
fn parse_package_name_version(text: &str) -> Option<(String, String)> {
    let json = serde_json::from_str::<serde_json::Value>(text).ok()?;
    let name = json.get("name")?.as_str()?.to_string();
    let version = json.get("version")?.as_str()?.to_string();
    Some((name, version))
}

fn json_object_key(trimmed: &str) -> Option<&str> {
    trimmed.strip_prefix('"')?.split('"').next()
}

/// Parses a JSON `"name": "spec"` line, returning the name, the column range of
/// the name (inside its quotes), the spec, and the column range of the spec
/// value (inside its quotes) within `line`.
fn parse_json_entry(line: &str) -> Option<(String, Range<u32>, String, Range<u32>)> {
    let key_open = line.find('"')?;
    let after_key_open = key_open + 1;
    let key_len = line[after_key_open..].find('"')?;
    let name = line[after_key_open..after_key_open + key_len].to_string();
    let name_cols = after_key_open as u32..(after_key_open + key_len) as u32;

    let after_key = after_key_open + key_len + 1;
    let colon = line[after_key..].find(':')?;
    let after_colon = after_key + colon + 1;
    let value_open = line[after_colon..].find('"')?;
    let value_start = after_colon + value_open + 1;
    let value_len = line[value_start..].find('"')?;
    let spec = line[value_start..value_start + value_len].to_string();

    Some((
        name,
        name_cols,
        spec,
        value_start as u32..(value_start + value_len) as u32,
    ))
}

/// Extracts the key from a YAML `key:` / `key: value` line, unquoting if needed.
fn yaml_key(trimmed: &str) -> Option<&str> {
    if trimmed.starts_with(['"', '\'']) {
        let quote = trimmed.as_bytes()[0] as char;
        let close = trimmed[1..].find(quote)? + 1;
        Some(&trimmed[1..close])
    } else {
        let colon = trimmed.find(':')?;
        Some(trimmed[..colon].trim_end())
    }
}

/// Parses a YAML `key: value` mapping line (used for catalog entries), returning
/// the key, the key's column range, the value, and the value's column range
/// within `line`. Returns `None` for keys with no value (e.g. a named-catalog
/// header `react17:`).
fn parse_yaml_entry(line: &str) -> Option<(String, Range<u32>, String, Range<u32>)> {
    let indent = line.len() - line.trim_start().len();
    let trimmed = &line[indent..];
    if trimmed.starts_with('#') || trimmed.starts_with('-') {
        return None;
    }

    let (name, name_cols, key_end) = if trimmed.starts_with(['"', '\'']) {
        let quote = trimmed.as_bytes()[0] as char;
        let close = trimmed[1..].find(quote)? + 1;
        let name = trimmed[1..close].to_string();
        let start = (indent + 1) as u32;
        (name.clone(), start..start + name.len() as u32, close + 1)
    } else {
        let colon = trimmed.find(':')?;
        let name = trimmed[..colon].trim_end().to_string();
        let start = indent as u32;
        (name.clone(), start..start + name.len() as u32, colon)
    };

    let after_key = &trimmed[key_end..];
    let colon = after_key.find(':')?;
    let after_colon = &after_key[colon + 1..];
    let leading_ws = after_colon.len() - after_colon.trim_start().len();
    let mut value = after_colon.trim_start();
    // Strip a trailing ` # comment`.
    if let Some((before, _)) = value.split_once(" #") {
        value = before;
    }
    let value = value.trim_end();
    if value.is_empty() {
        return None;
    }

    let value_offset = indent + key_end + colon + 1 + leading_ws;
    let (spec, spec_offset, spec_len) = if value.starts_with(['"', '\'']) {
        let quote = value.as_bytes()[0] as char;
        let close = value[1..].find(quote)?;
        (value[1..1 + close].to_string(), value_offset + 1, close)
    } else {
        (value.to_string(), value_offset, value.len())
    };
    Some((
        name,
        name_cols,
        spec,
        spec_offset as u32..(spec_offset + spec_len) as u32,
    ))
}

/// Extracts the base semver from a spec like `^1.2.3`, `~1.0`, `1.x`. Returns
/// `None` for pnpm/non-registry protocols (`workspace:`, `catalog:`, `npm:`,
/// git/file/url) and unparseable specs.
fn spec_version(spec: &str) -> Option<Version> {
    let spec = spec.trim();
    if spec.is_empty() || spec == "*" || spec == "latest" || spec.contains(':') {
        return None;
    }
    let trimmed = spec.trim_start_matches(['^', '~', '>', '=', '<', 'v', ' ']);
    let token = trimmed.split_whitespace().next().unwrap_or(trimmed);
    let core = token.split(['-', '+']).next().unwrap_or(token);
    let mut parts: Vec<String> = core
        .split('.')
        .map(|part| match part {
            "x" | "X" | "*" => "0".to_string(),
            other => other.to_string(),
        })
        .collect();
    while parts.len() < 3 {
        parts.push("0".to_string());
    }
    Version::parse(&parts[..3].join(".")).ok()
}

async fn fetch_versions(http: Arc<dyn HttpClient>, package: &str) -> Vec<Version> {
    let Some(request) = Request::builder()
        .uri(format!("{REGISTRY}/{package}"))
        .header("Accept", "application/vnd.npm.install-v1+json")
        .body(AsyncBody::empty())
        .log_err()
    else {
        return Vec::new();
    };
    let Some(mut response) = http.send(request).await.log_err() else {
        return Vec::new();
    };
    if !response.status().is_success() {
        return Vec::new();
    }
    let mut body = String::new();
    if response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .log_err()
        .is_none()
    {
        return Vec::new();
    }
    let Some(json) = serde_json::from_str::<serde_json::Value>(&body).log_err() else {
        return Vec::new();
    };
    let Some(versions) = json.get("versions").and_then(|v| v.as_object()) else {
        return Vec::new();
    };
    let mut versions: Vec<Version> = versions
        .keys()
        .filter_map(|key| Version::parse(key).ok())
        .filter(|version| version.pre.is_empty())
        .collect();
    versions.sort();
    versions
}

/// Builds the hover popover for the npm package under `anchor`, if any. Returns
/// the rendered markdown and the anchor range of the package name (for
/// highlighting). Detection is fast (and a no-op for non-dependency files); the
/// registry document is fetched once per package and cached.
pub(crate) async fn npm_package_hover(
    editor: &WeakEntity<Editor>,
    anchor: Anchor,
    cx: &mut AsyncWindowContext,
) -> Option<(String, Range<Anchor>)> {
    let (package, name_range, http) = editor
        .update(cx, |editor, cx| {
            let kind = editor.version_lens_kind(cx)?;
            let snapshot = editor.buffer().read(cx).snapshot(cx);
            let point = anchor.to_point(&snapshot);
            let (name, name_cols) =
                package_at_point(&snapshot.text(), kind, point.row, point.column)?;
            let range = snapshot.anchor_before(Point::new(point.row, name_cols.start))
                ..snapshot.anchor_after(Point::new(point.row, name_cols.end));
            Some((name, range, cx.http_client()))
        })
        .ok()
        .flatten()?;

    let cache = editor.update(cx, |_, cx| doc_cache(cx)).ok()?;
    let cached = cache.0.lock().unwrap().docs.get(&package).cloned();
    let doc = match cached {
        Some(DocState::Ready(doc)) => doc,
        Some(DocState::Failed) => return None,
        None => match fetch_package_doc(http, &package).await {
            Some(doc) => {
                let doc = Arc::new(doc);
                cache
                    .0
                    .lock()
                    .unwrap()
                    .docs
                    .insert(package.clone(), DocState::Ready(doc.clone()));
                doc
            }
            None => {
                cache
                    .0
                    .lock()
                    .unwrap()
                    .docs
                    .insert(package.clone(), DocState::Failed);
                return None;
            }
        },
    };

    Some((build_hover_markdown(&package, &doc), name_range))
}

/// Finds the dependency whose name or version value spans column `col` on `row`,
/// returning its package name and the name's column range.
fn package_at_point(
    text: &str,
    kind: LensFile,
    row: u32,
    col: u32,
) -> Option<(String, Range<u32>)> {
    parse_deps(text, kind).into_iter().find_map(|dep| {
        let hit = dep.row == row
            && (range_contains(&dep.name_cols, col) || range_contains(&dep.value_cols, col));
        hit.then_some((dep.name, dep.name_cols))
    })
}

fn range_contains(range: &Range<u32>, col: u32) -> bool {
    col >= range.start && col <= range.end
}

fn build_hover_markdown(name: &str, doc: &PackageDoc) -> String {
    let mut markdown = format!("## {name}\n\n");

    let mut meta = Vec::new();
    if let Some(latest) = &doc.latest {
        meta.push(format!("**{latest}** · latest"));
    }
    if let Some(license) = &doc.license {
        meta.push(license.clone());
    }
    if !meta.is_empty() {
        markdown.push_str(&meta.join(" · "));
        markdown.push_str("\n\n");
    }

    if let Some(description) = &doc.description {
        markdown.push_str(description);
        markdown.push_str("\n\n");
    }

    let mut links = vec![format!("[npm](https://www.npmjs.com/package/{name})")];
    if let Some(homepage) = &doc.homepage {
        links.push(format!("[homepage]({homepage})"));
    }
    if let Some(repository) = &doc.repository {
        links.push(format!("[repository]({repository})"));
    }
    markdown.push_str(&links.join(" · "));
    markdown.push_str("\n\n");

    if let Some(readme) = &doc.readme {
        markdown.push_str("---\n\n");
        markdown.push_str(readme);
    }
    markdown
}

async fn fetch_package_doc(http: Arc<dyn HttpClient>, package: &str) -> Option<PackageDoc> {
    let request = Request::builder()
        .uri(format!("{REGISTRY}/{package}"))
        .header("Accept", "application/json")
        .body(AsyncBody::empty())
        .log_err()?;
    let mut response = http.send(request).await.log_err()?;
    if !response.status().is_success() {
        return None;
    }
    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .log_err()?;
    let json = serde_json::from_str::<serde_json::Value>(&body).log_err()?;

    let latest = json
        .pointer("/dist-tags/latest")
        .and_then(|v| v.as_str())
        .and_then(|s| Version::parse(s).ok());
    let description = json
        .get("description")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let homepage = json
        .get("homepage")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let repository = json.get("repository").and_then(parse_repository_url);
    let license = json.get("license").and_then(parse_license);
    let mut readme = json
        .get("readme")
        .and_then(|v| v.as_str())
        .filter(|r| !r.trim().is_empty() && *r != "ERROR: No README data found!")
        .map(str::to_string);
    if let Some(readme) = &mut readme {
        if readme.len() > README_CAP {
            let mut end = README_CAP;
            while !readme.is_char_boundary(end) {
                end -= 1;
            }
            readme.truncate(end);
            readme.push_str("\n\n…");
        }
    }

    Some(PackageDoc {
        latest,
        description,
        homepage,
        repository,
        license,
        readme,
    })
}

/// Normalizes an npm `repository` field (string or `{ url }`) into a clickable
/// https URL.
fn parse_repository_url(value: &serde_json::Value) -> Option<String> {
    let raw = match value {
        serde_json::Value::String(url) => url.clone(),
        serde_json::Value::Object(object) => object.get("url")?.as_str()?.to_string(),
        _ => return None,
    };
    let url = raw.trim_start_matches("git+").trim_end_matches(".git");
    let url = if let Some(rest) = url.strip_prefix("git@github.com:") {
        format!("https://github.com/{rest}")
    } else if let Some(rest) = url.strip_prefix("git://") {
        format!("https://{rest}")
    } else if let Some(rest) = url.strip_prefix("github:") {
        format!("https://github.com/{rest}")
    } else {
        url.to_string()
    };
    Some(url)
}

/// Extracts an SPDX license id from the npm `license` field (string or the
/// legacy `{ type }` object form).
fn parse_license(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(license) => Some(license.clone()),
        serde_json::Value::Object(object) => {
            object.get("type").and_then(|v| v.as_str()).map(str::to_string)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(deps: &[Dependency]) -> Vec<&str> {
        deps.iter().map(|d| d.name.as_str()).collect()
    }

    #[test]
    fn test_parse_package_json() {
        let text = r#"{
  "name": "demo",
  "dependencies": {
    "react": "^18.0.0",
    "@scope/pkg": "1.2.3"
  },
  "scripts": {
    "build": "tsc"
  }
}"#;
        let deps = parse_package_json(text);
        assert_eq!(names(&deps), vec!["react", "@scope/pkg"]);
        assert_eq!(deps[0].spec, "^18.0.0");
        assert_eq!(deps[0].context, "dependencies");
        // The spec value range points at exactly the version text.
        let line = text.lines().nth(deps[0].row as usize).unwrap();
        assert_eq!(
            &line[deps[0].value_cols.start as usize..deps[0].value_cols.end as usize],
            "^18.0.0"
        );
    }

    #[test]
    fn test_parse_pnpm_catalogs() {
        let text = r#"packages:
  - "apps/*"

catalog:
  react: ^18.0.0
  "@types/node": ~20.0.0

catalogs:
  react17:
    react: ^17.0.0
"#;
        let deps = parse_pnpm_catalogs(text);
        assert_eq!(names(&deps), vec!["react", "@types/node", "react"]);
        assert_eq!(deps[0].context, "catalog");
        assert_eq!(deps[0].spec, "^18.0.0");
        assert_eq!(deps[1].name, "@types/node");
        assert_eq!(deps[1].spec, "~20.0.0");
        // The named-catalog `react` is distinguished by context.
        assert_eq!(deps[2].context, "catalogs.react17");
        assert_eq!(deps[2].spec, "^17.0.0");
        // Value range is correct (enables click-to-update).
        let line = text.lines().nth(deps[0].row as usize).unwrap();
        assert_eq!(
            &line[deps[0].value_cols.start as usize..deps[0].value_cols.end as usize],
            "^18.0.0"
        );
    }

    #[test]
    fn test_spec_version() {
        assert_eq!(spec_version("^1.2.3"), Some(Version::new(1, 2, 3)));
        assert_eq!(spec_version("~0.20.0"), Some(Version::new(0, 20, 0)));
        assert_eq!(spec_version("1.x"), Some(Version::new(1, 0, 0)));
        assert_eq!(spec_version("18"), Some(Version::new(18, 0, 0)));
        assert_eq!(spec_version("workspace:*"), None);
        assert_eq!(spec_version("catalog:"), None);
        assert_eq!(spec_version("*"), None);
    }

    #[test]
    fn test_compute_tiers() {
        let current = Version::new(18, 0, 0);
        let versions = [
            Version::new(18, 0, 5),
            Version::new(18, 3, 1),
            Version::new(19, 2, 7),
        ];
        let tiers = compute_tiers(&current, &versions);
        assert_eq!(tiers.patch, Some(Version::new(18, 0, 5)));
        assert_eq!(tiers.minor, Some(Version::new(18, 3, 1)));
        assert_eq!(tiers.major, Some(Version::new(19, 2, 7)));
        assert!(compute_tiers(&Version::new(19, 2, 7), &versions).is_empty());
    }

    #[test]
    fn test_parse_workspace_catalogs() {
        let text = r#"catalog:
  react: ^18.0.0
catalogs:
  react17:
    react: ^17.0.0
    react-dom: ^17.0.0
"#;
        let (default_catalog, named_catalogs) = parse_workspace_catalogs(text);
        assert_eq!(default_catalog.get("react"), Some(&"^18.0.0".to_string()));
        assert_eq!(
            named_catalogs.get("react17").and_then(|c| c.get("react")),
            Some(&"^17.0.0".to_string())
        );
        assert_eq!(
            named_catalogs
                .get("react17")
                .and_then(|c| c.get("react-dom")),
            Some(&"^17.0.0".to_string())
        );
    }

    #[test]
    fn test_resolve_ref() {
        let mut context = WorkspaceContext::default();
        context
            .default_catalog
            .insert("react".to_string(), "^18.0.0".to_string());
        context.named_catalogs.insert(
            "react17".to_string(),
            HashMap::from_iter([("react".to_string(), "^17.0.0".to_string())]),
        );
        context
            .local_versions
            .insert("@app/ui".to_string(), "1.4.0".to_string());

        assert_eq!(
            resolve_ref("catalog:", "react", Some(&context)).as_deref(),
            Some("catalog: → ^18.0.0")
        );
        assert_eq!(
            resolve_ref("catalog:react17", "react", Some(&context)).as_deref(),
            Some("catalog:react17 → ^17.0.0")
        );
        assert_eq!(
            resolve_ref("workspace:*", "@app/ui", Some(&context)).as_deref(),
            Some("workspace:* → 1.4.0")
        );
        // Unknown package / missing context resolve to nothing.
        assert_eq!(resolve_ref("catalog:", "vue", Some(&context)), None);
        assert_eq!(resolve_ref("catalog:", "react", None), None);
    }

    #[test]
    fn test_parse_package_name_version() {
        let text = r#"{ "name": "@app/ui", "version": "1.4.0", "private": true }"#;
        assert_eq!(
            parse_package_name_version(text),
            Some(("@app/ui".to_string(), "1.4.0".to_string()))
        );
        assert_eq!(parse_package_name_version(r#"{ "name": "x" }"#), None);
    }

    #[test]
    fn test_package_at_point() {
        let text = "{\n  \"dependencies\": {\n    \"react\": \"^18.0.0\"\n  }\n}";
        // The `react` key is on row 2; its name starts at column 5.
        let (name, name_cols) =
            package_at_point(text, LensFile::PackageJson, 2, 6).expect("hover over name");
        assert_eq!(name, "react");
        let line = text.lines().nth(2).unwrap();
        assert_eq!(
            &line[name_cols.start as usize..name_cols.end as usize],
            "react"
        );
        // Hovering the version value resolves to the same package.
        assert_eq!(
            package_at_point(text, LensFile::PackageJson, 2, 16).map(|(n, _)| n),
            Some("react".to_string())
        );
        // A blank column resolves to nothing.
        assert_eq!(package_at_point(text, LensFile::PackageJson, 0, 0), None);
    }

    #[test]
    fn test_parse_repository_url() {
        assert_eq!(
            parse_repository_url(&serde_json::json!({
                "type": "git",
                "url": "git+https://github.com/facebook/react.git"
            }))
            .as_deref(),
            Some("https://github.com/facebook/react")
        );
        assert_eq!(
            parse_repository_url(&serde_json::json!("git@github.com:user/repo.git")).as_deref(),
            Some("https://github.com/user/repo")
        );
    }

    #[test]
    fn test_build_hover_markdown() {
        let doc = PackageDoc {
            latest: Some(Version::new(19, 0, 0)),
            description: Some("A library".to_string()),
            homepage: Some("https://react.dev".to_string()),
            repository: Some("https://github.com/facebook/react".to_string()),
            license: Some("MIT".to_string()),
            readme: Some("# React\n\nHello".to_string()),
        };
        let markdown = build_hover_markdown("react", &doc);
        assert!(markdown.starts_with("## react"));
        assert!(markdown.contains("**19.0.0** · latest · MIT"));
        assert!(markdown.contains("A library"));
        assert!(markdown.contains("[npm](https://www.npmjs.com/package/react)"));
        assert!(markdown.contains("[homepage](https://react.dev)"));
        assert!(markdown.contains("# React\n\nHello"));
    }
}
