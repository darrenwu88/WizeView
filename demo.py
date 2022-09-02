import tkinter as tk
import mediapipe as mp
from PIL import ImageTk
from classifier import *
import timeit
class gui():
	def __init__(self):
		self.root = tk.Tk()
		self.root.configure(bg='white')
		self.root.title("fall assessment")
		self.root.geometry("1024x700")
		self.tab1()
		self.root.mainloop()
		self.mp_drawing = mp.solutions.drawing_utils



	def tab1(self):
		self.label1 = tk.Label(self.root, text="Welcome to Fall Assessment V1.13", font=("Arial", 30))
		self.label1.pack()
		self.label2 = tk.Label(self.root, text="Developed by Duke EGR101 Fall Assessment Team 1")
		self.label2.place(relx=0.5, rely=0.9, anchor='s')
		self.button1 = tk.Button(self.root, text="Start", command=self.tab2)
		self.button1.configure(height=2, width=10)
		self.button1.place(relx=0.5, rely=0.5, anchor="center")

	def tab2(self):
		self.label1.destroy()
		self.label2.destroy()
		self.button1.destroy()
		self.label3 = tk.Label(self.root, text="Instructions", font=("Arial", 30))
		self.label3.pack()
		self.label4 = tk.Label(self.root,
							   text="1. Place the camera 10 feet away from you such that your entire is seen in the frame.\n2. Pivot your chair 45 degrees from the camera",
							   )
		self.label4.pack()

		self.frame = tk.Frame(self.root, bg='white')
		self.frame.pack()
		self.video_label = tk.Label(self.frame)
		self.video_label.pack()
		self.cap = cv2.VideoCapture(0)
		self.video()
		self.button2 = tk.Button(self.root, text="Next", command=self.tab3)
		self.button2.configure(height=2, width=10)
		self.button2.pack()

	def video(self):
		_, self.frame = self.cap.read()
		cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)
		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)
		self.video_label.imgtk = imgtk
		self.video_label.configure(image=imgtk)
		self.video_label.after(1, self.video)

	def tab3(self):
		self.label3.destroy()
		self.label4.destroy()
		self.button2.destroy()
		self.label5 = tk.Label(self.root, text="Calibration", font=("Arial", 30))
		self.label5.pack()
		self.label6 = tk.Label(self.root,
							   text="We will calibrate this software by taking pictrues of you sitting down and standing up.\nHit the corresponding buttons when you are ready\n",
							   )
		self.label6.pack()
		self.button3 = tk.Button(self.root, text="Sit",command=self.sit_photo())
		self.button3.configure(height=2, width=10)
		self.button3.place(relx=1, rely=0.4, anchor="e")
		self.button4 = tk.Button(self.root, text="Stand",command = self.stand_photo())
		self.button4.configure(height=2, width=10)
		self.button4.place(relx=1, rely=0.6, anchor="e")
		self.button5 = tk.Button(self.root, text="START!!!", command=self.tab4)
		self.button5.configure(height=2, width=10)
		self.button5.pack()
		bootstrap_images_in_folder = 'fitness_poses_images_in'
		# Output folders for bootstrapped images and CSVs.
		bootstrap_images_out_folder = 'fitness_poses_images_out'
		bootstrap_csvs_out_folder = 'fitness_poses_csvs_out'

		# Initialize helper.
		bootstrap_helper = BootstrapHelper(
			images_in_folder=bootstrap_images_in_folder,
			images_out_folder=bootstrap_images_out_folder,
			csvs_out_folder=bootstrap_csvs_out_folder,
		)
		# Bootstrap all images.
		# Set limit to some small number for debug.
		bootstrap_helper.bootstrap()
		bootstrap_helper.align_images_and_csvs(print_removed_items=False)
		dump_for_the_app()

	def sit_photo(self):
		mp_drawing_styles = mp.solutions.drawing_styles
		mp_pose = mp.solutions.pose
		sit_count = 0
		with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
			while True:

				image = self.frame
				image.flags.writeable = False
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				results = pose.process(image)
				# Draw the pose annotation on the image.
				image.flags.writeable = True
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
				mp_drawing.draw_landmarks(
					image,
					results.pose_landmarks,
					mp_pose.POSE_CONNECTIONS,
					landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
				if sit_count < 5:
					# Flip the image horizontally for a selfie-view display.
					sit_count += 1
					cv2.imwrite("fitness_poses_images_in/sit/%d.jpg" % sit_count, self.frame)

	def stand_photo(self):
		mp_drawing_styles = mp.solutions.drawing_styles
		mp_pose = mp.solutions.pose
		stand_count = 0
		with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
			while True:
				image = self.frame
				image.flags.writeable = False
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				results = pose.process(image)
				# Draw the pose annotation on the image.
				image.flags.writeable = True
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
				mp_drawing.draw_landmarks(
					image,
					results.pose_landmarks,
					mp_pose.POSE_CONNECTIONS,
					landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
				if stand_count < 5:
					# Flip the image horizontally for a selfie-view display.
					stand_count += 1
					cv2.imwrite("fitness_poses_images_in/stand/%d.jpg" % stand_count, self.frame)
	def tab4(self):
		self.label5.destroy()
		self.label6.destroy()
		self.button3.destroy()
		self.button4.destroy()
		self.button5.destroy()
		timer = 0
		self.label7 = tk.Label(self.root,
							   text="Time Remaining: " + str(timer) ,
							   )
		self.label7.pack()
		class_name = 'stand'
		pose_samples_folder = 'fitness_poses_csvs_out'
		# Initialize embedder
		pose_embedder = FullBodyPoseEmbedder()

		# Initialize classifier.
		# Check that you are using the same parameters as during bootstrapping.
		pose_classifier = PoseClassifier(
			pose_samples_folder=pose_samples_folder,
			pose_embedder=pose_embedder,
			top_n_by_max_distance=30,
			top_n_by_mean_distance=10)

		# Initialize EMA smoothing.
		pose_classification_filter = EMADictSmoothing(
			window_size=10,
			alpha=0.2)

		# Initialize counter.
		repetition_counter = RepetitionCounter(
			class_name=class_name,
			enter_threshold=5.8,
			exit_threshold=4.5)

		pose_classification_visualizer = PoseClassificationVisualizer(
			class_name=class_name,
			# Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
		)

		pose_tracker = mp_pose.Pose()
		out_video_path = 'sit-stand-sample-out.mp4'
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
		video_width = 1024
		video_height = 768
		video_fps = self.cap.get(cv2.CAP_PROP_FPS)
		sit_count = 0
		stand_count = 0

		bootstrap_images_in_folder = 'fitness_poses_images_in'
		# Output folders for bootstrapped images and CSVs.
		bootstrap_images_out_folder = 'fitness_poses_images_out'
		bootstrap_csvs_out_folder = 'fitness_poses_csvs_out'

		# Initialize helper.
		bootstrap_helper = BootstrapHelper(
			images_in_folder=bootstrap_images_in_folder,
			images_out_folder=bootstrap_images_out_folder,
			csvs_out_folder=bootstrap_csvs_out_folder,
		)
		# Check how many pose classes and images for them are available.
		bootstrap_helper.print_images_in_statistics()
		# Bootstrap all images.
		# Set limit to some small number for debug.
		bootstrap_helper.bootstrap()
		# Check how many images were bootstrapped.
		# After initial bootstrapping images without detected poses were still saved in
		# the folderd (but not in the CSVs) for debug purpose. Let's remove them.
		bootstrap_helper.align_images_and_csvs(print_removed_items=False)
		bootstrap_helper.print_images_out_statistics()
		dump_for_the_app()
		# Open output video.
		out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps,
									(video_width, video_height))
		frame_idx = 0
		output_frame = None

		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
		video_width = 1024
		video_height = 768
		video_fps = self.cap.get(cv2.CAP_PROP_FPS)

		while True:
			if frame_idx%10==0:
				timer +=1
			# Get next frame of the video.
			success, input_frame = self.cap.read()
			if not success:
				break
			# Run pose tracker.
			input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
			result = pose_tracker.process(image=input_frame)
			pose_landmarks = result.pose_landmarks

			# Draw pose prediction.
			output_frame = input_frame.copy()
			if pose_landmarks is not None:
				self.mp_drawing.draw_landmarks(
					image=output_frame,
					landmark_list=pose_landmarks,
					connections=mp_pose.POSE_CONNECTIONS)
			544
			if pose_landmarks is not None:
				# Get landmarks.
				frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
				pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
										   for lmk in pose_landmarks.landmark], dtype=np.float32)
				assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

				# Classify the pose on the current frame.
				pose_classification = pose_classifier(pose_landmarks)

				# Smooth classification using EMA.
				pose_classification_filtered = pose_classification_filter(pose_classification)
				# Count repetitions.
				repetitions_count = repetition_counter(pose_classification_filtered)
			else:
				# No pose => no classification on current frame.
				pose_classification = None

				# Still add empty classification to the filter to maintaing correct
				# smoothing for future frames.
				pose_classification_filtered = pose_classification_filter(dict())
				pose_classification_filtered = None

				# Don't update the counter presuming that person is 'frozen'. Just
				# take the latest repetitions count.
				repetitions_count = repetition_counter.n_repeats

			# Draw classification plot and repetition counter.
			output_frame = pose_classification_visualizer(
				frame=output_frame,
				pose_classification=pose_classification,
				pose_classification_filtered=pose_classification_filtered,
				repetitions_count=repetitions_count,
				plot_x_max=frame_idx + 1)

			# Save the output frame.
			out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
			output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_BGR2RGB)
			cv2.imshow('MediaPipe Pose', output_frame)
			frame_idx += 1

			if cv2.waitKey(5) & 0xFF == ord("q"):
				break

		cv2.destroyAllWindows()
		self.cap.release()
		# Close output video.
		out_video.release()

		# Release MediaPipe resources.
		pose_tracker.close()

gui = gui()

