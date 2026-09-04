# Physical-Plausibility Validation Layer

Two independent RF-DETR segmentation models are run over the same photograph: one
predicts **damage** (6 classes), the other predicts **car parts** (19 classes).
This study asks whether the second can be used to audit the first — whether a damage
prediction can be rejected on the grounds that the part it lies on cannot
physically sustain that kind of damage.

A *tire flat* on a windscreen, a *glass shatter* on a bumper, a *lamp broken* on a
door: these are not merely low-confidence predictions, they are **anatomically
impossible** ones. The damage model has no access to part identity and cannot rule
them out. The parts model can.

---

## 📐 What is measured, and why not F1

The layer is a filter. It can only ever *remove* predictions, so it can never
raise recall, and F1 is the wrong instrument: a filter that removes 353 false
positives at the cost of 11 true positives is doing its job even when F1 falls,
because in claims assessment the two error types do not carry the same cost.

The reported quantities are therefore:

| quantity | question it answers |
|---|---|
| FP removed, FP cut % | how much of the false-positive burden is anatomically impossible |
| Δ precision | what the filter buys |
| Δ recall | what it costs |
| rejection precision | of everything the layer discards, what fraction deserved it |
| per-class impossibility rate | where the mechanism actually lives |

Every one of them is reported **across a confidence sweep**, not at a single
operating point. No claim in this study rests on one threshold.

---

## 📁 Files

```
dissertation/
├── physical_plausibility_layer.py   ← one script: the study, the recall-cost
│                                       audit, and the self-check suite
├── local_paths.example.json         ← copy to local_paths.json and edit
└── results/
    ├── plausibility_cardd.json      ← every table, both datasets
    ├── plausibility_hil.json
    └── self_check.json              ← 151 checks, 0 failures
```

**One script, one inference pass.** The recall-cost audit (TABLE 4) runs on the
predictions already in memory rather than as a separate program, so it costs
nothing extra, and the check suite lives in the same file as the code it checks
so the two cannot drift apart.

---

## ⚖️ The rule table

Parts are grouped into three sets:

| group | part classes |
|---|---|
| `GLASS_PARTS` | Front_Windshield_Glass, Front_Door_Glass, Rear_Door_Glass, Diggi_Back_Door_Glass |
| `LAMP_PARTS` | Headlight, Taillight |
| `TYRE_PARTS` | tyre |

Each damage class carries an *allowed* set (`None` = any named part) and an
*excluded* set:

| damage class | allowed on | excluded from | reasoning |
|---|---|---|---|
| crack | any | — | cracks occur on glass, plastic and paint alike; no anatomical rule exists |
| dent | any | glass, lamps, tyres | these deform by shattering or deflating, not by denting |
| glass shatter | glass only | — | only glazing shatters |
| lamp broken | lamps only | — | only a lamp housing can be a broken lamp |
| scratch | any | glass | see the limitation noted below |
| tire flat | tyres only | — | only a tyre can be flat |

`crack` is deliberately unconstrained. It is the internal control: a class the
layer *cannot* act on, so any difference measured on it would indicate a defect in
the harness rather than a finding.

### Attribution policies

For each prediction the layer computes, for every detected part, the fraction of
the *damage mask* lying on that part. Four policies convert those fractions into a
verdict, and all four are reported:

| policy | rule |
|---|---|
| `set` | the strict per-class reading over every part clearing `overlap_threshold` (0.20) |
| `dominant` | judge only the single largest part above that bar; abstain on a tie whose verdicts conflict |
| `majority` | area vote — do the permitted parts carry more of the mask than the forbidden ones? |
| `analytics` | any permitted part at the lower 0.15 bar — **the rule actually deployed in the application** (`MIN_ATTRIBUTION_OVERLAP = 0.15`) |

Verdicts are **three-valued**: `True` (plausible), `False` (impossible), `None`
(abstain — no part evidence clears the bar). **Only `False` rejects.** Abstention
is not rejection; a prediction the layer cannot judge is kept.

---

## 🗂️ Datasets

### CarDD — official test split

374 images, 785 ground-truth instances, **all 6 damage classes evaluated**.
Every instance carries exactly one damage label, so a prediction can match only
its own class.

| class | GT instances | segm AP |
|---|---:|---:|
| crack | 70 | 22.44 |
| dent | 236 | 37.38 |
| glass shatter | 71 | 94.73 |
| lamp broken | 69 | 75.58 |
| scratch | 307 | 33.97 |
| tire flat | 32 | 93.51 |

### Humans-in-the-Loop (HIL) — all annotated images

814 images, 6,482 ground-truth instances, **5 of 6 classes evaluated**.

- The published HIL folders are **swapped**: the directory named *Car parts
  dataset* is the one holding the damage annotations. The publisher's own
  description corroborates this — the release is 1,812 images split into 998
  car *parts* images and 814 car *damages* images, and the directory named
  *Car parts dataset* is the one containing 814. Group A of the self-check
  proves it independently by reading the labels in each directory.
- No HIL label corresponds to **tire flat**, so that class is not evaluated here
  and its predictions are discarded before scoring.
- HIL's `Broken part` label is accepted as **either** *glass shatter* **or** *lamp
  broken*. HIL does not distinguish them, so neither class's precision on this
  dataset is a clean measurement and the pair must be read as one merged class.
- `Corrosion`, `Flaking`, `Missing part` and `Paint chip` carry no meaning for this
  study and become **ignore regions**.

| source label | mapped to | instances |
|---|---|---:|
| Broken part | glass shatter \| lamp broken | 1,500 |
| Scratch | scratch | 3,242 |
| Dent | dent | 1,664 |
| Cracked | crack | 76 |

Ignore regions follow **COCO ordering**: detections are matched against real
ground truth *first*, and only then is an unmatched detection lying ≥ 0.50 inside
an ignore region excused as neither a hit nor an error. Applying the ignore test
before matching would delete predictions that would have matched a real instance
while leaving that instance in the recall denominator — on this dataset 51 of
6,482 scored instances lie wholly inside the ignore union, so the ordering is not
a hypothetical concern.

---

## 🧪 Method

1. Both models are run at native resolution 960, damage collected at confidence
   ≥ 0.05, parts at ≥ 0.45.
2. Part masks are **deduplicated** (NMS at IoU 0.30) and then **capped** to the
   number a car can have (2 headlights, 4 tyres, 1 roof, …), in that order —
   capping first would let two detections of the *same* headlight fill the quota
   and evict a genuine second one.
3. Every damage prediction is judged under all four policies.
4. Predictions judged `False` are removed; the base and layered arms are then
   scored independently by greedy matching at IoU ≥ 0.50.
5. **Paired image-level bootstrap**, 5,000 resamples, seed 0. Images are the
   resampling unit because predictions within one photograph share a parts
   segmentation and a viewpoint. Both arms are recomputed on the same resample,
   and **one shared resample matrix** is reused across every policy and every
   threshold, so all comparisons in every table below are paired.

Parts pre-processing removed **10** duplicate + **15** over-cap detections on
CarDD (median 3 parts/image; 19 images yielded no parts at all) and **76** + **194**
on HIL (median 13 parts/image; no image yielded zero parts). Where no part
evidence exists the layer abstains, so those 19 CarDD images pass through
untouched.

---

## 📊 Results — CarDD (policy `majority`)

| conf | preds | FP base → layer | FP cut % [95% CI] | Δ precision [95% CI] | Δ recall [95% CI] | rejection precision [95% CI] | rej FP / rej TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 10,609 | 9,873 → 9,520 | 3.58 [1.86, 5.97] | **+0.00139** [0.00059, 0.00243] | −0.01401 [−0.02596, −0.00464] | **0.970** [0.955, 0.985] | 353 / 11 |
| 0.10 | 4,167 | 3,464 → 3,357 | 3.09 [1.28, 5.92] | **+0.00220** [0.00026, 0.00503] | −0.01401 [−0.02596, −0.00464] | 0.907 [0.846, 0.948] | 107 / 11 |
| 0.20 | 1,700 | 1,045 → 1,019 | 2.49 [1.14, 4.28] | +0.00196 [−0.00067, 0.00469] | −0.01401 [−0.02596, −0.00464] | 0.703 [0.577, 0.844] | 26 / 11 |
| 0.30 | 1,086 | 491 → 477 | 2.85 [1.29, 4.76] | **+0.00381** [0.00001, 0.00810] | −0.01019 [−0.01880, −0.00275] | 0.636 [0.450, 0.833] | 14 / 8 |
| 0.40 | 789 | 247 → 236 | 4.45 [1.67, 7.96] | **+0.00696** [0.00048, 0.01482] | −0.00892 [−0.01733, −0.00247] | 0.611 [0.333, 0.870] | 11 / 7 |
| **0.50** | 617 | 128 → 124 | 3.13 [0.72, 6.36] | +0.00317 [−0.00146, 0.00896] | −0.00764 [−0.01423, −0.00243] | 0.400 [0.100, 0.750] | 4 / 6 |

**Bold** = 95% CI excludes zero.

Rejection precision is the number to read first. At confidence 0.05 the layer is
right about **97.0%** of what it discards; at 0.50 it is right about 40%. The
layer is not a fixed-quality filter — it is most valuable exactly where the
detector is weakest, because that is where anatomically impossible predictions
are abundant.

## 📊 Results — HIL (policy `majority`)

| conf | preds | FP base → layer | FP cut % [95% CI] | Δ precision [95% CI] | Δ recall [95% CI] | rejection precision [95% CI] | rej FP / rej TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 30,427 | 29,346 → 28,182 | 3.97 [3.54, 4.42] | +0.00013 [−0.00032, 0.00055] | −0.00602 [−0.00815, −0.00406] | 0.965 [0.953, 0.976] | 1,162 / 42 |
| 0.10 | 9,555 | 8,759 → 8,469 | 3.31 [2.79, 3.89] | +0.00093 [−0.00002, 0.00182] | −0.00262 [−0.00399, −0.00140] | 0.938 [0.909, 0.965] | 289 / 19 |
| 0.20 | 2,987 | 2,424 → 2,349 | 3.09 [2.26, 4.01] | **+0.00263** [0.00038, 0.00480] | −0.00123 [−0.00227, −0.00043] | 0.892 [0.809, 0.960] | 74 / 9 |
| 0.30 | 1,579 | 1,143 → 1,112 | 2.71 [1.81, 3.71] | **+0.00507** [0.00300, 0.00730] | −0.00015 [−0.00049, 0.00000] | 0.938 [0.838, 1.000] | 30 / 2 |
| 0.40 | 1,022 | 677 → 656 | 3.10 [1.91, 4.47] | **+0.00708** [0.00433, 0.01035] | **0.00000** [0.00000, 0.00000] | 0.952 [0.840, 1.000] | 20 / 1 |
| **0.50** | 720 | 446 → 436 | 2.24 [0.96, 3.77] | **+0.00536** [0.00230, 0.00909] | **0.00000** [0.00000, 0.00000] | **1.000** [1.000, 1.000] | 10 / 0 |

At and above confidence 0.40 the layer on HIL removes false positives and
destroys **nothing**: recall is unchanged with a degenerate confidence interval,
rejection precision is 1.000, and the precision gain is significant.

> **Note on counting.** `rejected TP` counts predictions discarded that had matched
> ground truth in the base arm. It can exceed `TP_base − TP_layer`, because matching
> is re-run on the surviving set and a freed ground-truth instance may then be
> claimed by a different prediction. On HIL at 0.05, 42 true positives were rejected
> but 3 of their instances were re-matched, so the net loss is 39.

### Policy comparison at confidence 0.50

| policy | CarDD Δ precision | CarDD Δ recall | CarDD rej FP/TP | HIL Δ precision | HIL rej FP/TP |
|---|---:|---:|---:|---:|---:|
| `set` | +0.00351 | −0.00637 | 4 / 5 | +0.00159 | 3 / 0 |
| `dominant` | +0.00351 | −0.00637 | 4 / 5 | **+0.00482** | 9 / 0 |
| `majority` | +0.00317 | −0.00764 | 4 / 6 | **+0.00536** | 10 / 0 |
| `analytics` | +0.00384 | −0.00510 | 4 / 4 | +0.00053 | 1 / 0 |

The deployed `analytics` rule is the most conservative: at the lower 0.15 bar a
single permitted part is enough to clear a prediction, so it rejects least and
costs least. `majority` is the most active. All four agree on direction
everywhere, which is the point of reporting them together — the finding does not
depend on which attribution rule is chosen.

---

## 🔍 Where the mechanism lives — per-class impossibility rate

At confidence 0.05, the fraction of judged predictions ruled anatomically
impossible, per class:

| class | CarDD rate % [95% CI] | judged | HIL rate % [95% CI] | judged |
|---|---:|---:|---:|---:|
| crack (*control, no rule*) | 0.00 [0.00, 0.00] | 1,816 | 0.00 [0.00, 0.00] | 4,815 |
| dent | 4.31 [1.69, 7.67] | 2,621 | 3.02 [2.44, 3.69] | 15,156 |
| **glass shatter** | **39.83** [29.60, 49.61] | 118 | **45.17** [40.26, 50.00] | 487 |
| **lamp broken** | **38.60** [30.26, 46.58] | 171 | **43.01** [39.17, 46.82] | 1,130 |
| scratch | 2.96 [0.19, 7.30] | 3,678 | 1.15 [0.80, 1.54] | 8,102 |
| **tire flat** | **26.85** [16.07, 38.18] | 108 | *not evaluable* | — |

The three tightly constrained classes are ruled impossible an order of magnitude
more often than the loosely constrained ones, and the pattern **replicates across
two independent datasets**. `crack`, which carries no rule, records exactly zero
on both — the harness is doing nothing it was not asked to do.

### What bounds the gain

Ranking the six CarDD classes by *rule restrictiveness* — the fraction of the 19
part classes each damage class is forbidden from — against their *per-class AP*
gives Spearman **ρ = +0.8286, exact p = 0.0583**. (This is the correlation the
run reports as `rho(restrictiveness, AP)`; restrictiveness against the
impossibility rates in the table above is a different pair, and a weaker one at
ρ = 0.7714.)

The positive sign is the finding, and it is an unwelcome one: **the rules are
most restrictive on the classes the detector already handles best.** Glass
shatter, lamp broken and tire flat are forbidden from 79%, 89% and 95% of the
part taxonomy respectively, and they are also the three classes the damage model
already scores highest on — 94.7, 75.6 and 93.5 AP. The classes where the
detector is weak, `crack` at 22.4 and `scratch` at 34.0, are the ones anatomy
barely constrains, because cracks and scratches genuinely can occur almost
anywhere on a car. The layer therefore has least authority exactly where the
detector needs the most help, and that — rather than any deficiency in the rules
— is what caps the aggregate gain.

The figure is reported as description, not as inference. With n = 6 classes the
smallest two-sided p an exact permutation test can return is 2/6! = **0.0028**,
and the observed ordering is one rank-swap from perfect, so a p of 0.0583 means
these six points cannot certify the relationship — not that there is none.
Restrictiveness and AP may also both be driven by the same underlying property,
namely how physically and visually distinctive a damage type is, and nothing in
six points can separate that from a causal reading. The per-class impossibility
rates above, whose confidence intervals are separated by a wide margin and which
replicate on a second dataset, carry the empirical weight.

---

## 🔬 The recall cost, instance by instance

The layer's entire statistically significant recall loss on CarDD at confidence
0.50 is **six instances** — 6/785 = 0.00764, exactly the measured Δ recall. Rather
than leave that as a number, **TABLE 4 of the run** enumerates them.

| # | image | damage | IoU with GT | parts beneath (fraction of damage mask) | cause |
|---|---|---|---:|---|---|
| 1 | `000204.jpg` | glass shatter | 0.986 | Rear_Door **1.0000**, Rear_Door_Glass **0.9999** | nested masks |
| 2 | `000288.jpg` | glass shatter | 0.816 | Rear_Door 0.9809, Rear_Door_Glass 0.9198 | nested masks |
| 3 | `003115.jpg` | glass shatter | 0.974 | Rear_Door 1.0000 *only* | parts-model miss |
| 4 | `003156.jpg` | glass shatter | 0.977 | Rear_Door 1.0000 *only* | parts-model miss |
| 5 | `003516.jpg` | lamp broken | 0.926 | Front_Windshield_Glass 1.0000 *only* | parts-model miss |
| 6 | `003842.jpg` | scratch | 0.728 | Front_Door **0.9921**, Front_Door_Glass **0.9921** | nested masks, exact tie |

Every one of the six matched real ground truth at IoU 0.73–0.99 with the correct
class. **The damage model was right all six times.** Two mechanisms account for
all of them, and both sit in the *parts* model:

- **Nested masks (rows 1, 2, 6).** `Rear_Door` is segmented as the whole door
  *including the window aperture*, and `Rear_Door_Glass` sits inside it, so damage
  on the window lies ~100% on both. The `majority` vote sums them as competing
  evidence when one in fact *contains* the other. In row 1 a correctly detected
  shattered window is discarded because the door it is set into covers one
  ten-thousandth more of the mask; row 6 is an exact tie, which `v > iv` resolves
  against the prediction.
- **Parts-model misses (rows 3, 4, 5).** The permitted part was never detected at
  all — two shattered windows where only the door was segmented, and one lamp
  covered by a windscreen mask.

**On HIL at the same threshold, under the same rules, the same code and the same
policy, the layer destroys zero true positives** across 814 images and 6,482
instances, including 3,242 scratches. That is the control which clears the rules
themselves: had the anatomical constraints been unsound, HIL — where the vehicle
is fully visible and in-domain for the parts model — is where the casualties would
appear. There are none.

A counterfactual re-audit with the scratch-on-glass exclusion relaxed recovers
exactly **one** of the six (row 6) — and that row is a nesting tie, not a
wiper-scratch case. The rule is therefore not the cause of the recall loss either.

**Conclusion.** The layer's recall cost is real, small, fully enumerable, and
attributable to parts-mask geometry rather than to the anatomical rules or to the
damage model. It is a parts-segmentation quality result.

---

## ⚠️ Threshold provenance and what is *not* claimed

- Confidence **0.50** is reported as the primary operating point because it is a
  conventional value fixed in advance. It was **not** selected on a held-out split,
  and **no claim rests on it alone** — every statistic above is reported across the
  full sweep, and the direction of the effect is stable across all six thresholds
  on both datasets.
- The two datasets are **not** independent evidence of the same magnitude. HIL is
  the parts model's own domain; CarDD is not. They agree on *direction* and on the
  *per-class pattern*, and they disagree on *cost* — which is itself the finding of
  the audit above.
- Neither dataset is a held-out selection set for the rule table. The rules were
  written from anatomy before these measurements, not tuned to them, and the rule
  table has not been changed in response to any number on this page.
- The `scratch` ⊄ `glass` exclusion is the one anatomically arguable entry —
  windscreens are routinely scratched by wiper blades. It is retained because the
  audit shows it costs one instance, and that instance is a mask-geometry artefact.
- Rejected false positives are partly rejected by the same nesting effect that
  costs the six true positives. Discarding them remains correct, but the mechanism
  is less clean than the rule table alone implies.
- `CAPS` describes a **single vehicle** while capping runs **per image**, so a
  photograph containing two cars could lose genuine parts. Neither CarDD nor HIL is
  a multi-vehicle set, but the caps would need to be per-car before this ran on
  street scenes.

### Known avenue, deliberately not taken

A containment-aware attribution rule — a permitted sub-part outranking the panel
that encloses it, since a window is *part of* a door and the two claims do not
compete — would recover rows 1, 2 and 6. It is **not** applied here. The claim
made here is about the layer as deployed, and amending the rule after seeing which six
instances it hurt would be selection on the test set. It would also reduce the
number of false positives caught, a trade that needs measuring rather than
assuming. It is recorded as future work.

---

## ✅ Verification

Every number on this page was produced by code that, when it was written, had
been checked against nothing except its own output. A silent error in a class
index, a mask decode or the matching loop would produce results that look
entirely reasonable and are entirely wrong. `--self_check` therefore tests the
parts that can fail quietly, and prints PASS or FAIL for each.

| group | what it establishes | checks |
|---|---|---:|
| A | the annotation file's category order really matches the constants the rules index by | 10 |
| B | decoded ground-truth masks agree with the stored areas and the actual image dimensions | 5 |
| C | rule logic, exhaustive over synthetic part contexts | 12 |
| D | matching, on hand-built cases with known answers | 7 |
| E | parts pre-processing — dedup-then-cap, including the case the ordering exists to protect | 7 |
| F | statistics — bootstrap, Spearman, exact permutation *p* | 6 |
| G | arithmetic invariants recomputed inside the result JSONs themselves | 104 |
| | **total** | **151** |

**All 151 pass.** Group G is the one that matters most for a reader who did not
run the code: it re-derives precision, recall, FP-cut percentage, per-class rates
and rejection precision from the raw `TP` / `FP` / `gt` counts stored in the JSON
and checks them against the published figures, verifies that every confidence
interval brackets its own point estimate, and confirms that predictions kept plus
predictions rejected equals predictions scored. Every table above is therefore
checked against the counts it was derived from, not merely reprinted.

Group H — which loads both checkpoints and verifies class-id ranges and mask
geometry against the live models — is opt-in, because it is the only group that
needs a GPU.

Group A deserves one specific note: it does not assume the HIL folders are
swapped, it **proves** it, by reading the labels in each directory and checking
which one actually contains damage annotations.

---

## ▶️ Reproduction

```bash
# every table, both datasets - TABLE 4 (the recall-cost audit) is included
python dissertation/physical_plausibility_layer.py --dataset cardd
python dissertation/physical_plausibility_layer.py --dataset hil

# the 151 self-checks; no GPU and no checkpoints needed
python dissertation/physical_plausibility_layer.py --self_check

# add group H, which loads both models and checks mask geometry
python dissertation/physical_plausibility_layer.py --self_check --with_models
```

Run the two datasets **before** `--self_check`: group G verifies arithmetic
invariants inside the result JSONs, so it needs them to exist.

**First, point the script at your data.** Copy `local_paths.example.json` to
`local_paths.json` and fill in the six paths — the two checkpoints, the CarDD
image directory and annotation file, and the two HIL roots. That file is not
tracked by git, because where your data sits is not part of any result. Every
value can also be given on the command line instead.

Defaults reproduce every number on this page: resolution 960, damage collection
threshold 0.05, parts threshold 0.45, part NMS IoU 0.30, `overlap_threshold` 0.20,
`attribution_overlap` 0.15, match IoU 0.50, ignore overlap 0.50, 5,000 bootstrap
resamples, seed 0, fp16. Checkpoint and dataset paths are the only arguments that
need to change.

The audit and the check suite call `judge()` and `greedy_match()` directly
rather than re-implementing them, so nothing that verifies a result can drift away
from the code that produced it.

---

## 🙏 Acknowledgements and citations

### Datasets

**CarDD** — the damage benchmark, used here as its official test split.

> X. Wang, W. Li and Z. Wu, "CarDD: A New Dataset for Vision-Based Car Damage
> Detection," *IEEE Transactions on Intelligent Transportation Systems*,
> vol. 24, no. 7, pp. 7202–7214, July 2023, doi: 10.1109/TITS.2023.3258480.

Project page: <https://cardd-ustc.github.io/>. Cite the IEEE version above
rather than the arXiv preprint (arXiv:2211.00945) — it is the same work, but the
journal reference is the citable one. Terms of use are stated on the project
page.

**Humans in the Loop — Car parts and car damages dataset**, used here as the
second evaluation arm. Released by Humans in the Loop under **CC0 1.0**
(public-domain dedication).

> Humans in the Loop, *Car parts and car damages dataset*.
> <https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/>

### Models

Both detectors are fine-tuned **RF-DETR** segmentation models (`RFDETRSegMedium`
for damage, `RFDETRSegNano` for parts), released by Roboflow under Apache-2.0.

> I. Robinson, P. Robicheaux, M. Popov, D. Ramanan and N. Peri, "RF-DETR: Neural
> Architecture Search for Real-Time Detection Transformers," arXiv:2511.09554,
> 2025. <https://github.com/roboflow/rf-detr>

This study uses the parts model only as a fixed, already-trained source of part
evidence, and evaluates nothing on the data it was trained on. Its training
corpus is nonetheless credited below, because seven of the eight sources are
CC BY 4.0 and attribution is a condition of that licence.

### Parts model training data

No single public dataset provides instance masks for every part class this
pipeline needs, so the parts model was trained on a composite assembled from
**eight public car-part segmentation datasets** — 11,547 images pooled, 7,708
retained after de-duplication and remapping onto a unified 19-class taxonomy.

| # | dataset | platform | images | classes | licence |
|---|---|---|---:|---:|---|
| 1 | Car Parts Dataset (DSMLR, IT-KMITL) | GitHub | 500 | 18 | citation requested |
| 2 | Car parts — *Segmentation* | Roboflow Universe | 1,755 | 9 | CC BY 4.0 |
| 3 | Car Parts Segmentation — *Person Detector* | Roboflow Universe | 603 | 19 | CC BY 4.0 |
| 4 | Car parts — *FleetBlox* | Roboflow Universe | 1,862 | 33 | CC BY 4.0 |
| 5 | car-parts — *Axion Technical Service* | Roboflow Universe | 819 | 30 | CC BY 4.0 |
| 6 | car parts — *Habibullah* | Roboflow Universe | 2,866 | 16 | CC BY 4.0 |
| 7 | car-seg — *Gianmarco Russo* | Roboflow Universe | 2,255 | 21 | CC BY 4.0 |
| 8 | car-parts — *Atheer Algarni* | Roboflow Universe | 887 | 20 | CC BY 4.0 |

The DSMLR dataset specifies no formal licence but requests citation of its
associated publication:

> K. Pasupa, P. Kittiworapanya, N. Hongngern and K. Woraratpanya, "Evaluation of
> deep learning algorithms for semantic segmentation of car parts," *Complex &
> Intelligent Systems*, 2021, pp. 1–13, doi: 10.1007/s40747-021-00397-8.
> Dataset: <https://github.com/dsmlr/Car-Parts-Segmentation>

The seven Roboflow Universe datasets, in table order:
[2](https://universe.roboflow.com/segmentation-9q8ob/car-parts-llqro) ·
[3](https://universe.roboflow.com/person-detector/car-parts-segmentation) ·
[4](https://universe.roboflow.com/fleetblox-car-damage/car-parts-bzaux) ·
[5](https://universe.roboflow.com/axion-technical-service-pvt-ltd/car-parts-xal6u) ·
[6](https://universe.roboflow.com/habibullah-hmpb8/car-parts-chf9t) ·
[7](https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm) ·
[8](https://universe.roboflow.com/atheer-algarni-gvico/car-parts-ypa1r)

The full source table and the label-to-taxonomy mapping are in the dissertation
report (Tables 6.3 and 6.4).

### Evaluation protocol

Average precision is computed with `pycocotools.cocoeval.COCOeval`, the reference
implementation of the COCO protocol.

> T.-Y. Lin *et al.*, "Microsoft COCO: Common Objects in Context," in
> *European Conference on Computer Vision (ECCV)*, 2014, pp. 740–755.
