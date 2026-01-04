# Palette Tutorial

This tutorial describes how to make a palette. To begin with, check out the
[default palette I made](./resource/palette_files/palette_default_consolab_fast.json) first.

# TODO: update tutorial

```json
{
  "name": "palette_default_consolab_fast",
  "templates": [
    {
      "layer": 0,
      "chars": " ",
      "font": "C:/Windows/Fonts/consolab.ttf",
      "font_size": 24,
      "char_bound_width": 13,
      "char_bound_height": 22,
      "match_method": "fast"
    },
    {
      "omitted": "the rest of contents have been omitted..."
    }
  ]
}
```

All templates should begin with a dictionary that has a property `name` and a 
property `templates`.

`name` is not functionally used in code, so technically you can name it anything you want. 

`templates` is more important. It's a list of dictionaries and each dictionary in it is
called a `layer`.

## Layer

`layer` property is not functionally used in code. It's there for you as a memo.

`chars` is a string that contains all characters you want to use in this layer (gradient level).
New line character and duplicated characters will be ignored.

`font` is the path that leads to your font. Make sure the font is installed on your PC and is there.
Moreover, you can use different fonts across different layers. Just make sure that you set the font
size, char_bound_width, char_bound_height values properly for each layer.

`font_size` is the size of the font, must be integer.

`char_bound_width` and `char_bound_height` are the size of one single character (pixels) in the image.

`approx_ratio` is used to approximate the comparison of template matching. Smaller value leads
to faster rendering but also degrades the matching quality. This value is completely useless if the
`match_method` is not `vector`, like in this case. This value must be between 0 and 1, with 0 
being exclusive and 1 being inclusive.

`vector_top_k` is used to filter the matching candidates by pixel amount. Smaller value leads
to faster rendering but also could miss retrieve better candidates. This value is completely useless if the
`match_method` is not `vector`, like in this case.

`match_method`: is the algorithm of template matching.

**match_method**

| code      | help                                                                                                                                |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------|
| slow      | The slowest matching algorithm with the best matching quality. The templates are grayscale.                                         |
| optimized | Almost twice as fast as slow. The templates are binary. The resulting image will look bold compared to slow method.                 |
| fast      | Almost twice as fast as optimized. Utilizes XOR comparison. The resulting image is very similar to optimized method.                |
| vector    | Almost ten times as fast as slow. Vectorize all smaller images and compare the flattened array. The resulting image is much bolder. |


🧑‍🏫 **My tips**: more layers will lead the algorithm to be more precise about the shading levels.
In theory, this will output better quality rendering if you do this correctly.
Usually for the layers in the higher rank you should use less dense characters. For example:
```text
. , '
```
For the layers in the lower rank you should use more dense characters. For example:
```text
% # @
```
The characters are not limited to ASCII characters. You can use other characters as long as 
your font supports them.
