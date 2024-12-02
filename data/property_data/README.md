### Property Data

______________________________________________________________________

This [prompt data](./prompt_data.csv) was generated as follows.

**Model**: [gpt-4o](https://platform.openai.com/docs/models#gpt-4o)

**Prompt**:

<blockquote>

1. In a single column, return 1000 distinct items
1. Add another column that numerically categorizes each item according to its size with respect to the reference points: ant, cat, person, jeep, stadium.
1. Add another column that numerically categorizes each item according to its physical phase with respect to the reference points: smoke, milk, wood, diamond.
1. Add another column that numerically categorizes each item according to its sentience with respect to the reference points: rock, tree, ant, cat, chimp, man.
1. Add another column that numerically categorizes each item according to its weight with respect to the reference points: watch, book, man, jeep, stadium.
1. Add another column that numerically categorizes each item according to its rigidity with respect to the reference points: water, skin, leather, wood, metal.
1. Add another column that numerically categorizes each item according to its temperature with respect to the reference points: ice, soup, fire, lava, sun.
1. Add another column that numerically categorizes each item according to its shape with respect to the reference points: square, sphere, ant, man, cloud.
1. Add another column that numerically categorizes each item according to its texture with respect to the reference points: glass, carpet, book, ant, man, sandpaper.
1. Add another column that numerically categorizes each item according to its opacity with respect to the reference points: glass, paper, cloud, man, wall.
1. Add another column that numerically categorizes each item according to its mobility with respect to the reference points: rock, tree, mushroom, animal, man, car.

Generate the data in CSV format. Write one item per line with the item and it's appropriate categories seperated by a commas.

</blockquote>

_Note_: the LLM had to be repeatedly prompted until about 1000 distinct items were generated. To maintain attention, the property attributes were generated in baches of 100 items.

#### Property Sub-Directories

Each of the sub-directories were auto-generated from the [raw prompt data](./prompt_data.csv) using [this script](../../scripts/generate_property_data.py)
