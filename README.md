# BEng-Final-Year-Project-Code-Position-Location-System
Over 25 years of GPS has shown it to be an incredibly useful tool allowing devices to locate
themselves within metres in outdoor environments. However due to the scattering of the signals by
the walls and other solid surfaces, locating devices inside of buildings or in urban areas using this
technique is far from reliable. This has led to the increase in research into indoor positioning systems
which use a range of technologies such as Wi-Fi, visible light, and acoustic methods to name a few.
This paper will outline the research and prototyping of an optical imaging based self-positioning
system which can be used in applications such as on robots for its navigation and tracking.
Initial research was placed into alternative systems to assess what had already been achieved and
what is possible before concentrating on understanding optical methods and how they could be
implemented. The basic concept of the system was to use a reference point detection system with a
trilateration or triangulation technique to find the final location of the system. The research showed
that a deep learning approach and a geometric circle intersection triangulation method was of most
suitable choice due to their practicality with the resources available, and their accuracy in reference
point and device positioning, respectively.
An overview of the system consists of a raspberry pi, a pi camera, custom and opensource software
and the numbers 1 to 6 each written individually on 6 A4 pieces of paper used as reference points.
The pi camera is directed at the ceiling with at least three reference points in its angle of view. Using
a combination of TensorFlow’s object detection API and OpenCV’s image processing tools the device
is then able to locate three reference points in each video frame using a bounding box round each.
The reference point location information is then used to locate both, the angle of the reference
points from the horizontal direction the device is facing and make an estimate of the distance
covered on the ceiling in the images horizontal. These respective pieces of information are then used
in a triangulation and scale factor equation to gather the X, Y and Z coordinates of the device in the
room it is situated.
The reference point detection system uses a premade supervised deep learning object detection
model produced by TensorFlow. To be able to detect the custom written reference points the model
required training – which involves affectively teaching the model what it needs to look for from
showing it a set of labelled images. After trying a range of premade datasets to train the model with
no success, a custom one was built. This consisted of a large set of manually labelled images of the
reference points in different positions on the ceiling. The result of training the model was very
successful with it being able to distinguish the reference points on the ceiling from a range of
distances and orientations of the camera.
The testing completed on the complete system consisted of the device being moved along a variety
of routes in the rooms horizontal X, Y plane, analysing the effect of changing orientation and location
under different sets of reference points. This was followed by some vertical depth tests – where the
device was placed at a range of heights in three orientations to see how well the z coordinate could
estimate its position. The final test carried out was a system performance test, assessing the speed
of the system when the device was both stationary and moving.
The outcome showed the system to have good level of accuracy with position metrics produced
having an average error of just under ±10cm. The system speed shows the results to be output
approximately every two seconds, showing there to be improvements to be made in this area to
bring it up to real-time reporting, suggestions of which are highlighted throughout the discussion
and future work sections. 
