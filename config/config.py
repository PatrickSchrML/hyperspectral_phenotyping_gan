BATCHSIZE = 32
CONTI_MEAN = 0.
CONTI_STD = 1.
N_CONTI = 3
N_DISCRETE = 1
NOISE = 10  # 10
NC = 3  # 5  # num classes
NDF = 160  # dim output signal
NGF = int(NDF / 4)
CUDA = True
NGPU = 1

model_path = "generated_leaf_infogan-n_classes{}-n_discrete{}-n_conti{}-n_noise{}".format(NC, N_DISCRETE, N_CONTI, NOISE)
NETG = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/{}/model{}/netG_epoch_{}{}.pth".format(model_path, "{}", "{}", "-crossval-0")  # "./model"
NETD = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/{}/model{}/netD_epoch_{}{}.pth".format(model_path, "{}", "{}", "-crossval-0")  # "./model"
OUTF = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/{}".format(model_path)
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

config_dict["NETG"] = NETG
config_dict["NETD"] = NETD
config_dict["OUTF"] = OUTF
config_dict["MANUALSEED"] = None
