from modsim2.similarity.metrics import mmd, otdd

ARGUMENTS = "arguments"
CLASS_KEY = "class"
METRIC_CLS_DICT = {
    "mmd": mmd.MMD,
    "otdd": otdd.OTDD,
}
