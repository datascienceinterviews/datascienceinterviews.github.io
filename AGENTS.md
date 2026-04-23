# AGENTS.md - AI Agent Instructions for dsprep.com

## Project Overview

This is **dsprep.com**, a MkDocs Material static site for data science interview preparation. It is owned and maintained by **SourceStrongAI** (sourcestrongai.com).

- **Site URL**: https://dsprep.com
- **Repo**: https://github.com/datascienceinterviews/datascienceinterviews.github.io
- **Framework**: MkDocs with Material theme
- **Contact email**: hi@sourcestrongai.com

## Local Development

- **Conda environment**: Use `conda run -n mk` for all mkdocs commands
- **Dev server**: `conda run -n mk mkdocs serve` (runs on http://127.0.0.1:8000)
- **Deploy**: `conda run -n mk mkdocs gh-deploy` (builds and pushes to `gh-pages` branch)
- **No CI/CD workflow** — deployment is manual via `mkdocs gh-deploy`

## Project Structure

```
mkdocs.yml                  # Site config, nav, plugins, theme
docs/                       # All site content (markdown, notebooks, assets)
  index.md                  # Homepage
  flashcards.md             # Flashcards app mount point
  contact.md                # Contact page
  privacy.md                # Privacy policy
  Contribute.md             # Contribution guide
  Interview-Questions/      # 15 topic files (markdown with Q&A format)
  Cheat-Sheets/             # Reference sheets (markdown + Jupyter notebooks)
  Machine-Learning/         # ML topic explainers
  Online-Material/          # External resource links
  javascripts/              # JS assets (flashcards.js, mathjax.js, xfile.js)
  stylesheets/extra.css     # Custom CSS overrides
overrides/main.html         # Template overrides (announcement bar, analytics, ads)
hooks/generate_questions_json.py  # Build hook: extracts Q&A to questions.json
site/                       # Built output (do not edit directly)
```

## Deployment

- **Source branch**: `master`
- **Deploy branch**: `gh-pages` (GitHub Pages serves from here)
- **CNAME**: `dsprep.com`
- After pushing to `master`, you must run `conda run -n mk mkdocs gh-deploy` to publish changes

## Interview Questions Format

Questions in `docs/Interview-Questions/*.md` follow this structure:

```markdown
### Question Title

**Difficulty:** <emoji> Level | **Tags:** `Tag1`, `Tag2` | **Asked by:** Company1, Company2

??? success "View Answer"

    Answer content (4-space indented)
    Supports markdown, code blocks, LaTeX math
```

The build hook (`hooks/generate_questions_json.py`) parses these on `on_post_build` and generates `site/assets/questions.json` (currently 915 questions) for the flashcards feature.

## Flashcards System

- **Page**: `docs/flashcards.md` — minimal HTML mount point
- **Logic**: `docs/javascripts/flashcards.js` — 800+ line client-side app
- **Data**: `site/assets/questions.json` — generated at build time by the hook
- Features: topic filtering (localStorage), card flip (CSS 3D), shuffle, MathJax support

## Announcement Bar

Located in `overrides/main.html`. Rotating slideshow cycling through:
- Flashcards feature promotion
- SourceStrongAI products (ProofMD.ai, ReplyRunner, SEO Audit, OnlineToolsVault, TrendingDraft)
- SourceStrongAI branding

## Analytics & Ads

- **Google Analytics**: GA4 property `G-CMC251LQT6` (configured in both `mkdocs.yml` and `overrides/main.html` gtag snippet)
- **Google AdSense**: `ca-pub-4988388949365963` (loaded in `overrides/main.html`)

## Key Conventions

- All contact references should use `hi@sourcestrongai.com`
- External product links should open in new tabs (`target="_blank" rel="noopener"`)
- The PDF export plugin (`with-pdf`) is disabled by default; enable with `ENABLE_PDF_EXPORT=1`
- Example/dummy emails in code samples (SQL, Python, etc.) should NOT be changed
- Copyright references SourceStrongAI with AGPL-3.0 license
