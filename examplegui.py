#!/usr/bin/env python

#Note check this out: http://johnroach.info/2011/03/02/image-capturing-from-webcam-using-opencv-and-pygame-in-python/

import pygtk, math, array
pygtk.require('2.0')
import gtk, gobject, cv, cairo
from PIL import Image

class VisualiseWidget(gtk.DrawingArea):
	def __init__(self):
		gtk.DrawingArea.__init__(self)
		self.image = None
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

	def SetImageByIpl(self,imIpl):
		assert imIpl.nChannels == 3

		#Convert IPL image to PIL image
		imgSize = cv.GetSize(imIpl)
		pilImg = Image.fromstring("RGB", cv.GetSize(imIpl), imIpl.tostring(), 'raw', "BGR")
		self.SetImageByPil(pilImg)

	def SetImageByPil(self, pilImg):
		
		#Convert PIL image to cairo surface
		pilRaw = array.array('B',pilImg.tostring("raw","BGRX",0,1))		
		stride = pilImg.size[0] * 4
		self.image = cairo.ImageSurface.create_for_data(pilRaw, cairo.FORMAT_RGB24, 
				pilImg.size[0], pilImg.size[1], stride)
		self.set_size_request(*pilImg.size)

class Base:
	def __init__(self):
		self.cap = cv.CaptureFromCAM(-1)
		capture_size = (320,200)
		cv.SetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_WIDTH, capture_size[0])
		cv.SetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_HEIGHT, capture_size[1])

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

		gobject.timeout_add(1000./25., self.UpdateImage)

	def delete_event(self, widget, event, data=None):
		print "delete event occurred"
		return False

	def destroy(self, widget, data=None):
		gtk.main_quit()

	def main(self):
		gtk.main()
		
	def UpdateImage(self):
		#print "x"
		#cv.GrabFrame(self.cap)
		#imIpl = cv.RetrieveFrame(self.cap)

		imIpl = cv.QueryFrame(self.cap)

		self.visArea.SetImageByIpl(imIpl)
		self.visArea.RedrawCanvas()
		return True


if __name__ == "__main__":
	base = Base()
	base.main()

