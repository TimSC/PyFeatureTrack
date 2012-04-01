#!/usr/bin/env python

#Note check this out: http://johnroach.info/2011/03/02/image-capturing-from-webcam-using-opencv-and-pygame-in-python/

import pygtk, math, array, numpy as np
pygtk.require('2.0')
import gtk, gobject, cv, cairo, opencv, multiprocessing, time, Queue
from PIL import Image

import klt, selectGoodFeatures, writeFeatures, trackFeatures

def IplToPilImg(imIpl):
	assert imIpl.nChannels == 3
	imgSize = cv.GetSize(imIpl)
	return Image.fromstring("RGB", cv.GetSize(imIpl), imIpl.tostring(), 'raw', "BGR")

class WebcamWidget(gtk.Invisible):

	def __init__(self):
		gtk.Invisible.__init__(self)

		self.toWorker, self.fromWorker = multiprocessing.Queue(), multiprocessing.Queue()
		self.buffer = []
		self.maxBufferSize = 100
		self.count = 0

		gobject.timeout_add(int(round(1000./50.)), self.UpdatePipe)
		self.p = multiprocessing.Process(target=self.PollCamera, args=(self.toWorker,self.fromWorker))
		self.p.start()

	def __del__(self):
		pass
		#self.Stop()

	def Stop(self):
		self.toWorker.put(("STOP",))
		self.p.terminate()

	def GetCurrentImg(self):
		if len(self.buffer) == 0:
			return None
		return self.buffer[-1]

	def GetFrameNum(self):
		return self.count

	def UpdatePipe(self):

		while not self.fromWorker.empty():
			try:	
				pipeData = self.fromWorker.get(0)
				ty = pipeData[0]
				if ty == "FRAME": 
					img = pipeData[1]
					self.buffer.append(img)
					self.count += 1
				while len(self.buffer) > self.maxBufferSize:
					self.buffer.pop(0)
				#print ty, len(self.buffer)
			except Queue.Empty:
				pass
			

		return True

	def PollCamera(self, toWorker, fromWorker):
		running = True
		cap = cv.CaptureFromCAM(-1)
		capture_size = (320,200)
		cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_WIDTH, capture_size[0])
		cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_HEIGHT, capture_size[1])

		#fps = cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FPS)

		while (running):
			try:
				pipeData = toWorker.get(0)
				#print "Worker",pipeData[0]
				if pipeData[0] == "STOP":
					running = False


			except Queue.Empty:
				pass

			imIpl = cv.QueryFrame(cap)
			if imIpl is not None:
				pilImg = IplToPilImg(imIpl)
			#	#print pilImg
				fromWorker.put(("FRAME",np.array(pilImg)))
			time.sleep(1./100.)

		fromWorker.put(("STOPPED",))	
		
		return True

class TrackingProcess:
	def __init__(self):
		self.currentTracking = []
		self.toWorker, self.fromWorker = multiprocessing.Queue(), multiprocessing.Queue()
		gobject.timeout_add(int(round(1000./50.)), self.UpdatePipe)
		self.p = multiprocessing.Process(target=self.Process, args=(self.toWorker,self.fromWorker))
		#self.p = multiprocessing.Process(target=Test, args=(pipeChile,))
		self.p.start()

	def __del__(self):
		pass

	def Stop(self):
		self.toWorker.put(("STOP",))
		self.p.terminate()

	def TrackFrame(self, frameArr):
		if self.toWorker.qsize() < 5:
			self.toWorker.put(("FRAME", frameArr))

	def GetCurrentTracking(self):
		return self.currentTracking

	def UpdatePipe(self):

		try:	
			pipeData = self.fromWorker.get(0)
			ty = pipeData[0]
			if ty == "TRACKING": 
				tr = pipeData[1]
				self.currentTracking = tr

		except Queue.Empty:
			pass

		return True

	def Process(self, toWorker,fromWorker):
		running = True
		currentFrame = None
		tc = klt.KLT_TrackingContext()
		fl = []
		prevImg = None

		while running:

			while not toWorker.empty():
				print toWorker.empty(), toWorker.qsize()
				try:
					pipeData = toWorker.get()
					if pipeData[0] == "STOP":
						running = False
					if pipeData[0] == "FRAME":					
						currentFrame = Image.fromarray(pipeData[1])
				except Queue.Empty:
					pass
				time.sleep(0.01)

			if currentFrame is not None and running:
				
				nFeatures = 50
				countActive = klt.KLTCountRemainingFeatures(fl)
				if countActive == 0 or prevImg is None:
					fl = selectGoodFeatures.KLTSelectGoodFeatures(tc, currentFrame, nFeatures)
				else:
					trackFeatures.KLTTrackFeatures(tc, prevImg, currentFrame, fl)
				#print fl
				fromWorker.put(("TRACKING",fl))
				prevImg = currentFrame
				currentFrame = None

			time.sleep(0.01)
	

class VisualiseWidget(gtk.DrawingArea):
	def __init__(self):
		gtk.DrawingArea.__init__(self)
		self.image = None
		self.trackerPos = []
		self.set_size_request(100,100)
		self.connect("expose_event", self.expose)
        
	def expose(self, widget, event):
		self.context = widget.window.cairo_create()
        
		# set a clip region for the expose event
		self.context.rectangle(event.area.x, event.area.y,
			event.area.width, event.area.height)
		self.context.clip()
        
		self.draw(self.context)
        
		return False

	def RedrawCanvas(self):
		self.redrawPending = False
		if self.window:
			alloc = self.get_allocation()
			self.queue_draw_area(0, 0, alloc.width, alloc.height)
			self.window.process_updates(True) 
   
	def draw(self, context):
		rect = self.get_allocation()
		x = rect.x + rect.width / 2
		y = rect.y + rect.height / 2

		if self.image is not None:
			context.set_source_surface(self.image, 
				0., 
				0.)
			context.paint()

		#Draw tracker points
		context.save()
		context.set_source_rgb(0, 1, 0) 
		for pt in self.trackerPos:
			context.arc(pt[0], pt[1], 2., 0. * math.pi, 2. * math.pi)
			context.fill()
		context.restore()

	def SetImageByIpl(self,imIpl):
		#Convert IPL image to PIL image
		#pilImg = opencv.adaptors.Ipl2PIL(imIpl)
		self.SetImageByPil(IplToPilImg(imIpl))
		
	def SetImageByPil(self, pilImg):
		
		#Convert PIL image to cairo surface
		pilRaw = array.array('B',pilImg.tostring("raw","BGRX",0,1))		
		stride = pilImg.size[0] * 4
		self.image = cairo.ImageSurface.create_for_data(pilRaw, cairo.FORMAT_RGB24, 
				pilImg.size[0], pilImg.size[1], stride)
		self.set_size_request(*pilImg.size)

class Base:
	def __init__(self):
		
		self.tc = klt.KLT_TrackingContext()
		self.fl = []
		self.webcam = WebcamWidget()
		self.showingFrame = None
		self.trackingProcess = TrackingProcess()

		#Create main window
		self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
		self.window.connect("delete_event", self.delete_event)
		self.window.connect("destroy", self.destroy)
		self.window.set_border_width(10)

		self.visArea = VisualiseWidget()
		#self.visArea.set_size_request((100,100))
		#self.visArea.set_from_file("img0.pgm")
		self.window.add(self.visArea)

		self.window.show_all()

		gobject.timeout_add(int(round(1000./25.)), self.UpdateImage)

	def delete_event(self, widget, event, data=None):
		self.webcam.Stop()
		self.trackingProcess.Stop()
		print "delete event occurred"
		return False

	def destroy(self, widget, data=None):
		print "destroy window"
		gtk.main_quit()

	def main(self):
		gtk.main()
		
	def UpdateImage(self):
		#print "x"
		#cv.GrabFrame(self.cap)
		#imIpl = cv.RetrieveFrame(self.cap)
		if self.showingFrame != self.webcam.GetFrameNum():
			img = self.webcam.GetCurrentImg()
			if img is not None:		
				self.visArea.SetImageByPil(Image.fromarray(img))

				currentTracking = self.trackingProcess.GetCurrentTracking()
				#print currentTracking
				self.visArea.trackerPos = []
				for pt in currentTracking:
					if pt.val < 0: continue
					self.visArea.trackerPos.append((pt.x,pt.y))

				self.visArea.RedrawCanvas()
				self.trackingProcess.TrackFrame(img)
			self.showingFrame = self.webcam.GetFrameNum()

		return True


if __name__ == "__main__":
	base = Base()
	base.main()

