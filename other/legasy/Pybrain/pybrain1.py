# from pybrain3.tools.shortcuts import buildNetwork
# from pybrain3.tools.xml import NetworkReader, NetworkWriter
# from pybrain3.datasets import SupervisedDataSet
# from pybrain3.supervised.trainers import BackpropTrainer
#
# import matplotlib.pyplot as plt
#
# ds = SupervisedDataSet(4, 1)
# ds.addSample([2, 3, 80, 1], [5])
# ds.addSample([5, 5, 50, 2], [4])
# ds.addSample([10, 7, 40, 3], [3])
# ds.addSample([15, 9, 20, 4], [2])
# ds.addSample([20, 11, 10, 5], [1])
#
# net = buildNetwork(4, 3, 1)  # hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True
# # print(f"Y1 = {net.activate([3, 7])}")
# # print(f"Y1 = {net.activate([-2, 1])}")
# # print(net)
# # print(net['bias'])
# # print(net['in'])
# # print(net['hidden0'])
# # print(net['out'])
#
# trainer = BackpropTrainer(net, dataset=ds, learningrate=0.01, momentum=0.1, weightdecay=0.01, verbose=True)
# trnerr, valerr = trainer.trainUntilConvergence()
# plt.plot(trnerr, 'b', valerr, 'r')
# plt.show()
