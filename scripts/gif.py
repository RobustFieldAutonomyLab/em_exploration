import imageio

images = []
for i in range(200):
    try:
        images.append(imageio.imread('step{}.png'.format(i)))
    except IOError:
        continue

imageio.mimsave('isrr2017_structured.gif', images, duration=0.5)
