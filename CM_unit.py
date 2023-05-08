from modeltraining.modelcompression import compress_model as cm

name='./saved_models/ARCH_2x37-REG_l1l2/checkpointPR_0.th'

cm.unit_test_seq(name)