# Artificial Art Generation with GANs
" Now the computers are some kinda painter themselves. "

`Note`: Due to lack of gpu resources this model can only generate low quality art/paintings on a 64x64 canvas

### Requirements
* PyTorch with torchvision
* Matplotlib / Numpy 

### Generate Random Art
1. Open the project directory in terminal/shell
2. Run the art generator script as -
> python art_with_gans.py
3. A random art will be saved on your current directory as 'generated_art.jpg'

### Training Process
The Generator model is Trained for 170 epochs and the improvement process was like:

<img src='https://github.com/VaibhavHaswani/Artificial_Art_Generation/blob/master/art_gans_training.gif'>
