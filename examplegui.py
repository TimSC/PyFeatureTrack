#!/usr/bin/env python

#Note check this out: http://johnroach.info/2011/03/02/image-capturing-from-webcam-using-opencv-and-pygame-in-python/

import pygtk, math, array, numpy as np
pygtk.require('2.0')
import gtk, gobject, cv, cairo, opencv, multiprocessing, time
from PIL import Image

import klt, selectGoodFeatures, writeFeatures, trackFeatures

def IplToPilImg(imIpl):
	assert imIpl.nChannels == 3
	imgSize = cv.GetSize(imIpl)
	return Image.fromstring("RGB", cv.GetSize(imIpl), imIpl.tostring(), 'raw', "BGR")

class WebcamWidget(gtk.Invisible):

	def __init__(self):
		gtk.Invisible.__init__(self)

		self.pipeParent, pipeChile = multiprocessing.Pipe()
		self.buffer = []
		self.maxBufferSize = 100
		self.count = 0

		gobject.timeout_add(int(round(1000./25.)), self.UpdatePipe)
		self.p = multiprocessing.Process(target=self.PollCamera, args=(pipeChile,))
		self.p.start()

	def __del__(self):
		pass
		#self.Stop()

	def Stop(self):
		self.pipeParent.send(("STOP",))
		self.p.terminate()

	def GetCurrentImg(self):
		if len(self.buffer) == 0:
			return None
		return self.buffer[-1]

	def GetFrameNum(self):
		return self.count

	def UpdatePipe(self):

		while self.pipeParent.poll():
			pipeData = self.pipeParent.recv()
			ty = pipeData[0]
			if ty == "FRAME": 
				img = pipeData[1]
				self.buffer.append(img)
				self.count += 1
			while len(self.buffer) > self.maxBufferSize:
				self.buffer.pop(0)
			#print ty, len(self.buffer)

		return True

	def PollCamera(self, pipe):
		running = True
		cap = cv.CaptureFromCAM(-1)
		capture_size = (320,200)
		cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_WIDTH, capture_size[0])
		cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_HEIGHT, capture_size[1])

		#fps = cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FPS)

		while (running):
			while pipe.poll():
				pipeData = pipe.recv()
				#print "Worker",pipeData[0]
				if pipeData[0] == "STOP":
					running = False
					continue

			imIpl = cv.QueryFrame(cap)
			if imIpl is not None:
				pilImg = IplToPilImg(imIpl)
				#print pilImg
				pipe.send(("FRAME",np.array(pilImg)))
			time.sleep(1./25.)

		pipe.send(("STOPPED",))	
		
		return True

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
				self.visArea.RedrawCanvas()
			self.showingFrame = self.webcam.GetFrameNum()

		return True

		imIpl = cv.QueryFrame(self.cap)
		
		pilImg = IplToPilImg(imIpl)
		if 0:
			nFeatures = 50
			countActive = klt.KLTCountRemainingFeatures(self.fl)
			if countActive == 0:
				self.fl = selectGoodFeatures.KLTSelectGoodFeatures(self.tc, pilImg, nFeatures)
			else:
				trackFeatures.KLTTrackFeatures(self.tc, self.prevImg, pilImg, self.fl)

			self.visArea.trackerPos = []
			for pt in self.fl:
				#print pt.x,pt.y,pt.val
				if pt.val < 0: continue
				self.visArea.trackerPos.append((pt.x,pt.y))

		self.visArea.SetImageByPil(pilImg)
		self.visArea.RedrawCanvas()

		self.prevImg = pilImg
		return True


if __name__ == "__main__":
	base = Base()
	base.main()

