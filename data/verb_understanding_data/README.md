### Verb Understanding Data

______________________________________________________________________

**Source**: https://github.com/google-deepmind/svo_probes

The [subject](./selectional_association_subject) and [object](./selectional_association_object) selectional association datasets were generated from the [svo probe data](./svo_probes.csv) using [this script](../../scripts/generate_selectional_association_data.py)

______________________________________________________________________

**Note**: The [pep_3k](./pep_3k) and [twentyquestions](./twentyquestions/) selectional association datasets were generated from the [pep_3k training data](../plausibility_data/pep_3k/valid.csv) and [twentyquestions training data](../plausibility_data/twentyquestions/valid.csv), respectively, using the same script. These auxilary datasets are used solely for [analyzing the relationship between selectional association and plausibility](../../scripts/run_selectional_association_plausibility_correlation.py)
