"""Published fixed train/validation partitions used by the pCE baseline.

These lists are intentionally kept in source control rather than sampled at
runtime.  They identify *subjects* (ACDC), *volumes* (MSCMR), or *scans*
(WORD).  The training script validates the complete list against the data on
disk, so a reorganized or incomplete dataset cannot silently change a result.
"""

# ScribFormer, ``acdc/dataset.py``, MAAGfold70.  The remaining 15 ACDC
# subjects are the official held-out test set.
ACDC_TRAIN = (37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26,
              69, 46, 59, 4, 89, 71, 6, 52, 43, 45, 63, 93, 14, 98, 88,
              21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40,
              7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42, 23, 92, 29,
              11, 83, 68, 75, 67, 16, 48, 66, 20, 15)
ACDC_VAL = (84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95)

# CycleMix's public ``data/MSCMR`` split. ScribbleBench removes subject2 and
# subject4 because their dense labels are unavailable; they must not be put
# back into a pCE experiment that validates against dense masks.
MSCMR_TRAIN = (13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 31, 32, 34,
               37, 39, 42, 44, 45, 6, 7, 9)
MSCMR_VAL = (1, 29, 36, 41, 8)

# Original WORD-V0.1.0 archive: ``imagesTr`` and ``imagesVal``. The archive
# is authoritative because ScribbleBench merges these two folders into
# imagesTr.  The 30 imagesTs IDs remain untouched as the official test split.
WORD_TRAIN = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 18, 20, 22, 26, 27, 28,
              29, 30, 32, 36, 38, 40, 41, 42, 44, 46, 47, 49, 51, 53, 55,
              56, 58, 59, 61, 62, 63, 64, 65, 67, 68, 70, 71, 72, 73, 78,
              79, 81, 82, 84, 86, 87, 89, 90, 91, 93, 94, 95, 96, 100, 101,
              102, 104, 105, 106, 107, 108, 109, 111, 113, 114, 115, 116,
              117, 118, 119, 121, 122, 123, 125, 126, 127, 128, 130, 132,
              133, 134, 135, 136, 138, 140, 142, 143, 144, 145, 146, 147,
              148, 150)
WORD_VAL = (1, 7, 15, 25, 31, 35, 39, 45, 48, 66, 75, 80, 83, 85, 98, 112,
            137, 139, 141, 149)

ACDC_TEST = (5, 10, 24, 30, 36, 39, 41, 60, 64, 72, 73, 77, 78, 80, 82)
MSCMR_TEST = (10, 11, 12, 16, 17, 23, 28, 30, 33, 35, 38, 3, 40, 43, 5)
WORD_TEST = (14, 16, 17, 19, 21, 23, 24, 33, 34, 37, 43, 50, 52, 54, 57,
             60, 69, 74, 76, 77, 88, 92, 97, 99, 103, 110, 120, 124, 129, 131)


def published_groups(dataset_name):
    """Return (train group names, validation group names, protocol metadata)."""
    if dataset_name == "ACDC":
        make_name = lambda value: "patient{:03d}".format(value)
        return (
            tuple(map(make_name, ACDC_TRAIN)), tuple(map(make_name, ACDC_VAL)),
            "ScribFormer MAAGfold70 (70/15/15 patients; ED and ES together)",
        )
    if dataset_name == "MSCMR":
        make_name = lambda value: "subject{}".format(value)
        return (
            tuple(map(make_name, MSCMR_TRAIN)), tuple(map(make_name, MSCMR_VAL)),
            "CycleMix public split (23/5/15 after removing subject2/subject4 without dense GT)",
        )
    if dataset_name == "WORD":
        make_name = lambda value: "word_{:04d}".format(value)
        return (
            tuple(map(make_name, WORD_TRAIN)), tuple(map(make_name, WORD_VAL)),
            "WORD-V0.1.0 official folders (100/20/30 scans)",
        )
    raise ValueError("No published split registered for {}".format(dataset_name))


def published_test_groups(dataset_name):
    """Return fixed test IDs for checking the official test partition."""
    if dataset_name == "ACDC":
        return tuple("patient{:03d}".format(value) for value in ACDC_TEST)
    if dataset_name == "MSCMR":
        return tuple("subject{}".format(value) for value in MSCMR_TEST)
    if dataset_name == "WORD":
        return tuple("word_{:04d}".format(value) for value in WORD_TEST)
    raise ValueError("No published split registered for {}".format(dataset_name))
