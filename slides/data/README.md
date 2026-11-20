# Chart data and provenance

These files contain the values shown in the deck. Pilot summaries are from the linked [metamer-sae notebooks](https://github.com/neyedhayo/metamer-sae/tree/fd17f03c4feda64c1d4a3cb55d5bdab7a3cd6c8f/notebooks). The chart build reads the CSVs listed in the first table and the JSON demo records directly.

| File | Slide | Meaning |
| --- | --- | --- |
| `pca-variance.csv` | 10 | Cumulative fraction of natural-image training variance explained. |
| `topk-illustration.csv` | 11 | Illustrative latent scores for the TopK slider. |
| `sae-validation-sweep.csv` and `sae-selected-checkpoints.csv` | 13 | Natural-image validation mean squared error, sweep means and checkpoint metadata. |
| `latent-1157-source-activations.csv` and `latent-1157-condition-means.csv` | 14 | Seed-123 development-set activations for latent 1157. Source image scores are in the first file. The second gives condition means across all eight development triplets. |
| `pilot-paired-results.csv` | 16–17 | Mean L2 and active-support Jaccard by representation and condition, paired differences, and 95% bootstrap intervals. The table gives the sign convention per row. |
| `pilot-cosine-gain.csv` and `pilot-spatial-relative-l2.csv` | 18 | Paired cosine gain (adversarial minus standard), and mean relative L2 on raw spatial maps. The cosine range shows the 10th to 90th percentiles. |
| `pilot-reconstruction-delta.csv` | 19 | Metamer minus source reconstruction MSE, scaled by 1,000 as shown on the chart, with 95% bootstrap intervals. |

## Demo records and image data

- `generator.json` supplies the error trace, frame IDs and channels for the 180-step presentation replay on slide 4.
- `observer.json` supplies the channels for the visualization on slide 9 and the worked triplet distances on slide 16. The observer is frozen ConvNeXt-Tiny Stage 3.
- `observer-pooled-vectors.csv` contains the corresponding channel means for triplets `0000` and `0116`. `observer-0000-spatial-channels.csv` contains the displayed spatial channel values for triplet `0000`. Row, column and channel indices are zero based.
- `reconstructions.json` supplies the per-image MSEs and color scales for slide 12. `reconstruction-demo-channels.csv` has the original centered activations and rebuilt channel values. The maps reflow channels 0–383 in row-major order into 12 rows and 32 columns. The PCA and TopK fits use the same 1,800 natural-image training examples. The SAE checkpoint was selected on 420 validation images. Values are from the local presentation fit.
- `stimulus-sources.csv` identifies the Feather release files converted losslessly to WebP under `assets/stimuli/`.

The heatmap image files are the rendered views of the numeric records. Channel CSVs are provided for inspection and replotting.
