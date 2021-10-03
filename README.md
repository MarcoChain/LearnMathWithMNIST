# Learn math with MNIST  <a href="https://pytorch.org/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a>  </p>


  ![alt text](https://github.com/MarcoChain/LearnMathWithMNIST/blob/main/images/plus.png?raw=true)

  ![alt text](https://github.com/MarcoChain/LearnMathWithMNIST/blob/main/images/minus.png?raw=true)


In this project we will develop a very naive variational auto-encoder (VAE) to perform very simple math operation. To learn more about this argument I strongly suggest to follow the  [`Yann LeCun's course at NYU`](https://www.youtube.com/watch?v=fs0wacmh_mI).


```python
class Calculator(nn.Module):
  def __init__(self):
    super(Calculator, self).__init__()
    self.encoder = nn.Sequential(
        nn.Linear(784*3, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256,20)
    )
    self.decoder = nn.Sequential(
        nn.Linear(10, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Sigmoid())
```

Even without convolutional layers, the result obtained is quite cool. The model learns how to compute addition and subtraction in the range [0-9]. This notebook is inspired by the [Yannic](https://www.youtube.com/watch?v=udS2OPohs_s) youtube video and his [notebook](https://colab.research.google.com/drive/1tDjFW7CFGQG8vHdUAVNpr2EG9z0JZGYC?usp=sharing).

