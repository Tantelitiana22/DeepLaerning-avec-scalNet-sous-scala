import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.core.Dense
import org.deeplearning4j.scalnet.models.Sequential
import org.deeplearning4j.scalnet.regularizers.L2
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

object testDL4J {

  private val logger: Logger = LoggerFactory.getLogger(testDL4J.getClass)
  def main(args: Array[String]) = {

    val height: Int = 28
    val width: Int = 28
    val nClasses: Int = 10
    val batchSize: Int = 64
    val hiddenSize = 512
    val seed: Int = 123
    val epochs: Int = 15
    val learningRate: Double = 0.0015
    val decay: Double = 0.005
    val scoreFrequency = 1000

    val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, seed)
    val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, seed)

    logger.info("Build model...")
    val model: Sequential = Sequential(rngSeed = seed)

    model.add(Dense(hiddenSize, height * width, activation = Activation.RELU, regularizer = L2(learningRate * decay)))
    model.add(Dense(hiddenSize, activation = Activation.RELU, regularizer = L2(learningRate * decay)))
    model.add(Dense(nClasses, activation = Activation.SOFTMAX, regularizer = L2(learningRate * decay)))
    model.compile(LossFunction.NEGATIVELOGLIKELIHOOD)

    logger.info("Train model...")
    model.fit(mnistTrain, epochs, List(new ScoreIterationListener(scoreFrequency)))


    logger.info("Evaluate model...")
    println(s"Train accuracy = ${model.evaluate(mnistTrain).accuracy}")
    println(s"Test accuracy = ${model.evaluate(mnistTest).accuracy}")

  }
}
