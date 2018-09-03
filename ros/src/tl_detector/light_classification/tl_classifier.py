from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
import cv2
import time

THRESHOLD_SCORE = 0.09

class TLClassifier(object):
    def __init__(self , is_site ):
		self.visualize = False
		self.processing_frame = False
		self.counter = 0
		self.graph = tf.Graph()
		self.class_dic = {1 : 'Red' , 2 : 'Yellow' , 3 : 'Green'}
		self.color_dic = { 0: (255,255,255) , 1 : (0,0,255) , 2 : (0,255,255) , 3 : (0,255,0)}

		if is_site:
		    model_name = "model_real.pb"
		else :
		    model_name = "model_sim.pb"

		with tf.Session(graph=self.graph) as sess:
		    rospy.logwarn(model_name)
		    self.sess = sess
		    od_graph_def = tf.GraphDef()
		    od_graph_def.ParseFromString(tf.gfile.GFile('light_classification/' + model_name , 'rb').read())
		    tf.import_graph_def(od_graph_def, name='')

		    self.bounding_boxes_tensor     = self.graph.get_tensor_by_name('detection_boxes:0')
		    self.predicted_score_tensor    = self.graph.get_tensor_by_name('detection_scores:0')
		    self.predicted_classes_tensor  = self.graph.get_tensor_by_name('detection_classes:0')
		    self.predicted_obj_num_tensor  = self.graph.get_tensor_by_name('num_detections:0')
		    self.input_tensor              = self.graph.get_tensor_by_name('image_tensor:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.processing_frame:
            rospy.logwarn('recived new frame before processing last one droping new frame')
            return
    
        self.processing_frame = True

        t0 = time.time()
        image = np.array(image)

        with self.graph.as_default():
            if not self.visualize:
                (scores, classes) = self.sess.run([self.predicted_score_tensor, self.predicted_classes_tensor],
                                       {self.input_tensor: [image]})
            else :
                (boxes , scores, classes) = self.sess.run(
                            [self.bounding_boxes_tensor ,self.predicted_score_tensor, self.predicted_classes_tensor],
                                       {self.input_tensor: [image]})
            #rospy.logwarn(classes)
            #rospy.logwarn(scores)
            rospy.logwarn(time.time() - t0)
            if self.visualize:
                self.save_image(image , boxes[0] , scores[0], classes[0] )

            #rospy.logwarn(classes[0])
            #rospy.logwarn(scores[0])
            for detected_class , score in zip(classes[0] , scores[0]):
                if int(detected_class) == 1 and score > THRESHOLD_SCORE :
                    rospy.logwarn('***************************RED************************')
                    self.processing_frame = False
                    return TrafficLight.RED

        self.processing_frame = False
        return TrafficLight.UNKNOWN

    def save_image(self , image , boxes , scores , classes):
        img_shape = image.shape

        for i in range(len(scores)):
            score = scores[i]
            box = boxes[i]
            detected_class = classes[i]

            if score > THRESHOLD_SCORE :
                cv2.rectangle(image , (int(box[1] * img_shape[1]) , int(box[0] * img_shape[0]) ) , ( int(box[3] * img_shape[1]) , int(box[2] * img_shape[0]) ) , self.color_dic[int(detected_class)]
                                , 4)


            
        cv2.imwrite('./output_images/img_' + str(self.counter) + '.jpg' , image)
        self.counter += 1
        