BATCHSIZE = 32
CONTI_MEAN = 0.
CONTI_STD = 1.
N_CONTI = 2
N_DISCRETE = 1
NOISE = 100  # 10
NC = 4  # 5  # num classes
NDF = 160  # dim output signal
NGF = int(NDF / 4)
CUDA = True
NGPU = 1

NETG_completion = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/generated_leaf_infogan/model/netG_epoch_39999.pth"
NETD_completion = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/generated_leaf_infogan/model/netD_epoch_39999.pth"

NETG_generate = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/generated_leaf_infogan_{}{}/model{}/netG_epoch_{}{}.pth".format(
    NC, "-crossval", "{}", "{}", "-crossval-0")  # 3 cluster
NETD_classify = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/generated_leaf_infogan_{}{}/model/netD_epoch_40000{}.pth".format(
    NC, "-crossval", "-crossval-0")

NETG = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/generated_leaf_infogan_{}{}/model{}/netG_epoch_{}{}.pth".format(
    NC, "-crossval", "{}", "{}", "-crossval-0")  # "./model"
NETD = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/generated_leaf_infogan_{}{}/model{}/netD_epoch_{}{}.pth".format(
    NC, "-crossval", "{}", "{}", "-crossval-0")  # "./model"
OUTF = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/generated_leaf_infogan_{}{}".format(NC,
                                                                                                                    "-crossval")
MANUALSEED = None

config_dict = dict()
config_dict["BATCHSIZE"] = BATCHSIZE
config_dict["CONTI_MEAN"] = CONTI_MEAN
config_dict["CONTI_STD"] = CONTI_STD
config_dict["N_CONTI"] = N_CONTI
config_dict["N_DISCRETE"] = N_DISCRETE
config_dict["NOISE"] = NOISE
config_dict["NC"] = NC
config_dict["NDF"] = NDF
config_dict["NGF"] = NGF
config_dict["CUDA"] = CUDA
config_dict["NGPU"] = NGPU

config_dict["NETG_completion"] = NETG_completion
config_dict["NETD_completion"] = NETD_completion

config_dict["NETG_generate"] = NETG_generate
config_dict["NETD_classify"] = NETD_classify

config_dict["NETG"] = NETD_classify
config_dict["NETD"] = NETD
config_dict["OUTF"] = OUTF
config_dict["MANUALSEED"] = None
