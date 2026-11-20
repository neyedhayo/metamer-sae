# What Does a Model Ignore?

Samuel Oyeneye · Visual Perception Project 08

## Present offline

Open **`index.html`** in a modern browser. The 25-slide deck embeds its images, charts, styles and scripts. The slide tools and chart-data download work offline. Citation links open their sources when you have internet access.

- **← / →** or **Page Up / Page Down** move between slides.
- **O** opens the slide overview. **N** opens speaker notes. **F** toggles fullscreen.
- The browser’s **Print → Save as PDF** prints one slide per page.

## Edit and rebuild

Use Python 3.10 or newer. The build uses the standard library and all its inputs are in this folder.

```sh
python3 build.py
```

On Windows, use `py -3 build.py`. The command regenerates `index.html`.

| File or folder | What to edit |
| --- | --- |
| `src/slides.html` | Slide copy, order, layouts, source links and speaker notes. Each slide is one `<section class="slide">`. |
| `src/settings.json` | Deck title and cover title markup. |
| `src/deck.css` and `src/deck.js` | Visual styles and interactions. |
| `src/figures.py` | Chart layout, axes and data-driven demo markup. |
| `data/` | Editable chart tables and demo records. See [data provenance](data/README.md). |
| `assets/` | Local stimuli and rendered demo frames and maps. |

To add a slide, copy a simple section in `src/slides.html` and give it a unique `id` and a new `data-title`. Move the section to set the order. To add an image, drop it anywhere under `assets/` and use its unique filename stem.

```html
<img data-photo="my-image" alt="Description of the image">
```

The build inserts the charts and demos at the double-braced placeholders. When updating results, edit the CSV values and review the slide text, speaker notes and axis ranges for any corresponding changes. For a heatmap update, replace the image and its accompanying numeric data together.

## Add to metamer-sae

Copy this folder into your repository, for example as `presentation/`. Commit the source, assets, data and rebuilt `index.html` together. The HTML file can also be shared by itself for presenting offline.

## Credits

- Published visual stimuli and model metamer procedure · [Feather et al. 2023](https://doi.org/10.1038/s41593-023-01442-0). [Stimulus and code release](https://github.com/jenellefeather/model_metamers_pytorch). The [stimulus manifest](data/stimulus-sources.csv) links each included image to its original PNG.
- Pilot measurements · [metamer-sae notebooks at the deck revision](https://github.com/neyedhayo/metamer-sae/tree/fd17f03c4feda64c1d4a3cb55d5bdab7a3cd6c8f/notebooks).
- The generation replay and reconstruction heatmaps were computed for the deck. The reconstruction comparison uses a local refit on an Imagenette subset. The pilot reconstruction chart uses the recorded study results.
