Context: SphinxLens v9 training is done. Before touching the backend,
you need to know about THREE class transmutations we made in the dataset.

A "transmutation" here means: an existing class index in the 150-class
head was RENAMED to represent a different Gardiner sign, because the
original sign had too few instances to ever be learnable, while the new
sign was common in our corpus/photos and was previously being mislabeled
as "unknown".

The three transmutations (index -> old sign -> new sign):

  idx 30 : E10 (goat)              -> M4  (time palm / year ideogram)
  idx 61 : I15 (coiled serpent)    -> F34 (heart)
  idx 70 : M20 (papyrus thicket)   -> O29 (wooden column)

Important: this is NOT a new class added (head is still fixed at nc=150).
It's a re-labeling of an existing output index. No architecture change.

New artifacts from v9 training (in artifacts/):
  best_model_v9.onnx
  class_map50_v9.json               <- names are CLEANED (no "000_" prefixes)
  confusion_matrix_v9_normalized.csv
  confusion_matrix_v9_raw.csv
  confusion_matrix_v9.npy

Please update:
  - Settings.onnx_path      -> best_model_v9.onnx
  - Settings.class_map      -> class_map50_v9.json
  - Settings.confusion_csv  -> confusion_matrix_v9_normalized.csv
  - sphinx_trie.py / GARDINER_MAP unchanged (lexicon doesn't depend on
    this YOLO version)

Validation metrics (162-image multi-glyph val set, matches production
conditions):
  Global    : mAP50=0.925  mAP50-95=0.699  P=0.914  R=0.874
  cartouche : mAP50=0.920  R=0.796  (up from v4's 0.885 / 0.783)
  f34 (ex i15) : mAP50=0.789  R=0.604
  o29 (ex m20) : mAP50=0.703  R=0.541
  m4  (ex e10) : mAP50=0.604  R=0.279  <-- weakest of the three, watch this
                                            in real-world testing

Sanity check before deploying: confirm class_map50_v9.json has 150 clean
names (no numeric prefixes) and that index 30/61/70 map to m4/f34/o29.

If you wonder why I made "class transmutation" instead of just adding 4 new classes yo my dataset,  the anser is it , my 
yolo model head is fixed for 150 classes from v1 era
if I were to add new classes +3 or +4 classes
that will reuin the work from months,,  
add new classes to yolo model  , it would have forced restart from scratch 
and all my fine tuning  curriculum learnign will be lost 
it has been a multple steps learning from v1

if you wonder why it is version 9 instead of v5, 
last realibel version was v4 training , our current model in ./artifacts
it becuase v5 ,v6,v7,v8 trainign were  corrupted,  doom and rubbish made
v5:v8 were useless models
hence I train version 9 model on a bigger dataset 
557 images
18840 instance
150 fixed classes
new transmutated classes are
e10 -> m4 now
i15 -> f34 
m20 -> o29

if you want to know why i choose this clases
you can run  python check_bbaw_corpus_freq.py 
you will find out that e10, i15, m20 are rare classes. 
hence f34, m3, o29 are more freq classes on bbaw corpus.


