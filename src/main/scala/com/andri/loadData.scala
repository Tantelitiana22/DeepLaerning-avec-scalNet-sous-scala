package com.andri

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator

object loadData {
    def loadData(batchSize:Int,seed:Int): (MnistDataSetIterator, MnistDataSetIterator) ={
      val mnistTrain = new MnistDataSetIterator(batchSize, true, seed)
      val mnistTest = new MnistDataSetIterator(batchSize, false, seed)

      (mnistTrain,mnistTest)
    }
}
