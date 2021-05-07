def default_augs(client):
    return [[]]


def func_1(client):
    return [
        [(client.LogoTranslation, [-0.6, 0.6])],
        [(client.LogoRotationZ, [-180, 180])],
        [(client.LogoRotationXY, [-45, 45])],
        [(client.LogoScale, [0.2, 1])],
    ]


def single_best_params(client):
    return [
        [(client.LogoTranslation, [-0.125, 0.125])],
        [(client.LogoRotationZ, [-160, 160])],
        [(client.LogoRotationXY, [-160, 160])],
        [(client.LogoScale, [0.2, 1])],
    ]


def single_best_params_v2(client):
    return [
        [(client.LogoTranslation, [-0.125, 0.125])],
        [(client.LogoRotationZ, [-160, 160])],
        [(client.LogoRotationXY, [-70, 70])],
        [(client.LogoScale, [0.2, 1])],
    ]


def single_easy_params(client):
    return [
        [(client.LogoTranslation, [-0.125, 0.125])],
        [(client.LogoRotationZ, [-30, 30])],
        [(client.LogoRotationXY, [-40, 40])],
        [(client.LogoScale, [0.2, 1])],
    ]


augmentations = {
    "default": default_augs,
    "test": func_1,
    "all_single_best": single_best_params,
    "all_single_best_v2": single_best_params_v2,
    "single_easy_params": single_easy_params,
    # "all_subset":
}
