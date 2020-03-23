#detection of face with mtcnn on a photo
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
 
#drawing object around the detected face
def draw_image_with_boxes(filename, result_list):
	#image loader
	data = pyplot.imread(filename)
	#plottwer
	pyplot.imshow(data)
	ax = pyplot.gca()
	#drwaing boxes
	for result in result_list:
		#coordinates
		x, y, width, height = result['box']
		#shaper
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		#box drawing
		ax.add_patch(rect)
		#identifing the eyes, nose and mouth
		for key, value in result['keypoints'].items():
			dot = Circle(value, radius=5, color='orange')
			ax.add_patch(dot)
	#displaying
	pyplot.show()
 
filename = 'peoples.jpg'
#load image from file
pixels = pyplot.imread(filename)
#creating the detector, using default weights
detector = MTCNN()
#detecting faces
faces = detector.detect_faces(pixels)
#display images with face identified
draw_image_with_boxes(filename, faces)