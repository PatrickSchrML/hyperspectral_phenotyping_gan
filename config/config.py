BATCHSIZE = 32
CONTI_MEAN = 0.
CONTI_STD = 1.
N_CONTI = 3
N_DISCRETE = 1
NOISE = 10 # 10
NC = 3  # 5  # num classes
NDF = 160  # dim output signal
NGF = int(NDF / 4)
CUDA = True
NGPU = 1

FROM_DATASET = True

NETG_completion = "/home/patrick/repositories/hsgan_new/trained_models/generated_leaf_infogan/model/netG_epoch_39999.pth"
NETD_completion = "/home/patrick/repositories/hsgan_new/trained_models/generated_leaf_infogan/model/netD_epoch_39999.pth"

NETG_generate = "/home/patrick/repositories/hsgan_new/trained_models/generated_leaf_infogan_{}{}/model{}/netG_epoch_{}{}.pth".format(NC, "-crossval" if FROM_DATASET else "", "{}", "{}", "-crossval-0" if FROM_DATASET else "")  # 3 cluster
#NETG_generate = "/home/patrick/repositories/hyperspec/used_methods/hsgan/generated_leaf_infogan_semisup_{}{}/model/netG_epoch_{}{}.pth".format(NC, "-crossval" if FROM_DATASET else "", "{}", "-crossval-0" if FROM_DATASET else "")  # 8 cluster
NETD_classify = "/home/patrick/repositories/hsgan_new/trained_models/generated_leaf_infogan_{}{}/model/netD_epoch_40000{}.pth".format(NC, "-crossval" if FROM_DATASET else "", "-crossval-0" if FROM_DATASET else "")

EPOCH_PRETRAINED = 12000
NETG = "/home/patrick/repositories/hsgan_new/trained_models/generated_leaf_infogan_{}{}/model{}/netG_epoch_{}{}.pth".format(NC, "-crossval" if FROM_DATASET else "", "{}", EPOCH_PRETRAINED, "-crossval-0" if FROM_DATASET else "")  # "./model"
NETD = "/home/patrick/repositories/hsgan_new/trained_models/generated_leaf_infogan_{}{}/model{}/netD_epoch_{}{}.pth".format(NC, "-crossval" if FROM_DATASET else "", "{}", EPOCH_PRETRAINED, "-crossval-0" if FROM_DATASET else "")  # "./model"
OUTF = "/home/patrick/repositories/hsgan_new/trained_models/generated_leaf_infogan_{}{}".format(NC, "-crossval" if FROM_DATASET else "")
MANUALSEED = None
