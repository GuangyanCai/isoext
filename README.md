diff-voxel
================

|      CI              | status |
|----------------------|--------|
| pip builds           | [![Pip Action Status][actions-pip-badge]][actions-pip-link] |
| wheels               | [![Wheel Action Status][actions-wheels-badge]][actions-wheels-link] |

[actions-pip-link]:        https://github.com/GuangyanCai/diff_voxel/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/GuangyanCai/diff_voxel/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/GuangyanCai/diff_voxel/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/GuangyanCai/diff_voxel/workflows/Wheels/badge.svg

Installation
------------

1. Clone this repository
2. Run `pip install ./diff_voxel`

Afterwards, you should be able to issue the following commands (shown in an
interactive Python session):

```pycon
>>> import diff_voxel
>>> diff_voxel.add(1, 2)
3
```