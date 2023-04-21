from modsim2.similarity import metrics

ARGUMENTS = "arguments"
CLASS_KEY = "class"
METRIC_CLS_DICT = {
    "mmd": metrics.MMD,
    "otdd": metrics.OTDD,
}
