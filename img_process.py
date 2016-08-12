# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 20:03:58 2016

@author: ziheng
"""

from __future__ import division
import numpy as np

_IMG_USE_OPENCV = False

from skimage import io
from skimage import transform


 

def img_read(img_dir, channel='rgb'):
  '''Read an image and store it into a ch*h*w ndarray
  
  The image will be in rgb space by default.
  
  Args
  -------
  img_dir
    Path to the image to be read
  channel
    'rgb' for rgb color space and 'bgr' for gbr color space
  
  Returns
  -------
  A processed ndarray image
  '''  
  if _IMG_USE_OPENCV:
    img = cv2.imread(img_dir)
    if channel == 'rgb':
      img = img[:,:,[2,1,0]]
    return img
  else:
    img = io.imread(img_dir)
    if channel == 'bgr':
      img = img[:,:,[2,1,0]]
    return img
  
  
def img_val(img, coord, val = None):
  ''' Read or set the pixel value at coordinate (x, y)
  
  Args
  -------
  img: The input image
  coord: The coordinate (x, y) willing to read or set
  val: The value willing to set at coordinate (x, y), val = None means read
  
  Returns
  -------
  The pixel value of input image at coordinate 
  '''
  if val != None:
    img[coord[1], coord[0], :] = val
    
  return img[coord[1], coord[0], :]
    
def img_norm(img, mean = 0, std = 1):
  '''Normalize an input image
  
  Normalize the image with mean and std
  
  Args
  -------
  img
    An image stored in a h*w*ch numpy ndarray
  mean
    Specify the desire mean to be substract from img
  std
    Specify the desire standard deviation
      
  Returns
  -------
  A normalized image
  '''

  if mean == None:
    output =  img - img.mean(axis=(0,1), keepdims=True)
    assert(std == None)
    output = output / output.std(axis=(0,1), keepdims=True)
  else:
    output =  (img - mean) / std

  return output
 

def img_tocaffeblob(img):
  ''' Rearranges the dimensions of img to match caffe's requirement
  
  The dimentions of the image will be rearranged as ch*h*w.
  
  Args
  -------
  img
    Input image(s) as ndarray
  bs
    Batch size
      
  Returns
  -------
  A processed ndarray image
  '''
  
  output = img.transpose(2, 0, 1)
    
  return output
  
def img_fromcaffeblob(blob, idx = 0):
  ''' Read an image from the caffe blob
  
  Args
  -------
  blob
    Caffe data blob
  idx
    Which image in the blob will be read
    
  Returns
  -------
  A ndarray image
  '''
  if blob.ndim == 4:
    img = blob[idx,...]
  else: img = blob
  img = img.transpose(1, 2, 0)
  return img
  
def img_scale(img, f = (1, 1), d = None, val = 0):
  '''Scale an image by a factor or to a desired size
  
  Args
  -------
  img
    The input image
  f
    Scale the image by factor f = (fh, fw)
  d
    Scale the image to a desired size (h, w)
  val
    The padding value
    
  Returns
  -------
  A processed ndarray image
  '''
  if _IMG_USE_OPENCV:  
    if f != None:
      dh, dw = int(img.shape[0]*f[0]), int(img.shape[1]*f[1])
    if d != None:
      dh, dw = d
      output = cv2.resize(img, (dw, dh))      
  else:
    if d is not None:
      f = d[0] / img.shape[0], d[1] / img.shape[1]
    output = transform.rescale(img, f, cval = val, preserve_range = True)
    output = output.astype(img.dtype)
  return output

def img_crop(img, roi):
  ''' Crop an image
  
  Only reserve the roi region of the whole image
  
  Args
  -------
  img
    The input image
  roi
    The region of interest, i.e the reserved region, as a rectangular
    [x_start, y_start, x_end, y_end]
      
  Returns
  -------
  The Cropped image
  '''
  output = img[roi[1]:roi[3]+1,roi[0]:roi[2]+1,...]
      
  return output
  
def img_cropas(img, d):
  ''' Crop an image to the desired shape
  
  Args
  -------
  img
    The input image
  d
    The desired output shape
    
  Returns
  -------
  The Cropped image
  '''
  c_err = max(0, ((img.shape[0] - d[0])) / 2.0), max(0, ((img.shape[1] - d[1]) / 2.0))
  x_start = int(np.floor(c_err[1]))
  x_end = int(max(0, img.shape[1] - np.ceil(c_err[1]) - 1))
  y_start = int(np.floor(c_err[0]))
  y_end = int(max(0, img.shape[0] - np.ceil(c_err[0]) - 1))
  
  return img_crop(img, [x_start, y_start, x_end, y_end])
  
def img_croproias(img, roi, d = None, val = 0):
  ''' 
  Crop the roi from img and scale to the desired shape d, blank area will
  be padded with value val
  
  Args
  -------
  img
    The input image
  roi
    The region of interest as (x0, y0, x1, y1)
  d
    The desired output size as (h, w)
  val
    The padding value
  '''
  output = img_crop(img, roi)
  if d is not None:
    output = img_shapeas(output, d, val = val, mode = 'scale')
  return output

def img_pad(img, l, r, t, b, val = 0):
  ''' Pad an image
  
  Args
  -------
  img
    The input image
  l, r, t, b
    Left, Right, Top, Bottom padding size
  val
    Pad the image with the value val
  
  Returns
  -------
  Padded image
  '''
  if len(img.shape) == 3:
    p_l = np.ones((img.shape[0], l, img.shape[2]))*val
    p_r = np.ones((img.shape[0], r, img.shape[2]))*val
    p_t = np.ones((t, img.shape[1]+l+r, img.shape[2]))*val
    p_b = np.ones((b, img.shape[1]+l+r, img.shape[2]))*val
  elif len(img.shape) == 2:
    p_l = np.ones((img.shape[0], l))*val
    p_r = np.ones((img.shape[0], r))*val
    p_t = np.ones((t, img.shape[1]+l+r))*val
    p_b = np.ones((b, img.shape[1]+l+r))*val
  output = np.concatenate((p_l, img, p_r), axis=1).astype(img.dtype)
  output = np.concatenate((p_t, output, p_b), axis=0).astype(img.dtype)

  return output

def img_padas(img, d, val = 0):
  ''' Pad around the image to shape d
  
  Args
  -------
  img
    The input image
  d
    Desired output shape d = (h, w)
    
  Returns
  -------
  Padded image
  '''

  c_err = max(0, ((d[0] - img.shape[0]) / 2.0)), max(0, ((d[1] - img.shape[1]) / 2.0))
  l = int(np.floor(c_err[1]))
  r = int(np.ceil(c_err[1]))
  t = int(np.floor(c_err[0]))
  b = int(np.ceil(c_err[0]))
  
  return img_pad(img, l, r, t, b, val = val)

def img_shapeas(img, d, val = 0, mode = 'crop'):
  ''' Pad or Crop the image to the desired size
  
  Args
  -------
  img
    The input image
  d
    Desired output shape d = (h, w)
  val
    Pad the image with value val if necessary
  mode: 'crop' or 'scale'
    In 'crop' mode, the area beyond the desired size will be cropped, and the
    center area will be reserved, the blank area will be padded with value val.
    
    In 'scale' mode, the image will be scaled so that the image is no larger
    than the desired size, the blank area will be padded with value val.
  
  Returns
  -------
  Processed image
  '''
  if mode == 'crop':
    output = img_cropas(img, d)
  elif mode == 'scale':
    f = min(float(d[0])/img.shape[0], float(d[1])/img.shape[1])
    output = img_scale(img, (f, f), val = val)
  output = img_padas(output, d, val = val)
  
  return output
  
def img_rot(img, angle, center = None, rescale = False, val = 0):
  ''' Rotate the image
  d
  Args
  -------
  img : ndarray
    Input image
  angle : float
    Rotation angle
  center : array-like
    The desired rotation center center = (x, y)
  rescale : bool
    Specify should the image be scaled to match the original size
  val
    The padding value
    
  Returns
  -------
  The Rotated image
  '''
  if _IMG_USE_OPENCV:
    d = None
    w = img.shape[1]
    h = img.shape[0]
    rangle = np.deg2rad(angle) # angle in radians
    if d == None:
      # now calculate new image width and height
      nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))
      nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))
    else:
      nh, nw = d
      
    scale = min(nw / (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w)), 
                nh / (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w)))
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(img, rot_mat, (int(np.ceil(nw)), int(np.ceil(nh))), 
                          borderValue = (val, val, val),flags=cv2.INTER_LANCZOS4)
  else:
    return transform.rotate(img, angle, center = center, cval = val,
                            resize = rescale, preserve_range = True).astype(img.dtype)
                        
def img_flip(img, direction = 'lr'):
  ''' Flip an image
  
  Args
  -------
  img : ndarray
    The input image
  direction : string
    Along which direction to flip, 'ud' for up-down flip and 'lr' for left-right
    flip
    
  Returns
  -------
  The flipped image
    
  '''
  if _IMG_USE_OPENCV:
    if direction == 'lr':
      return cv2.flip(img, 1)
    elif direction == 'ud':
      return cv2.flip(img, 0)
    else:
      raise ValueError('Invalid direction value!')
  else:
    if direction == 'lr':
      return np.fliplr(img)
    elif direction == 'ud':
      return np.flipud(img)
    else:
      raise ValueError('Invalid direction value!')
    
def img_centeras(img, c):
  '''Make pixel c=(h, w) as the center of the image
  
  Circular shift a image so that c=(h, w) become the center of the image
  
  Args
  -------
  img : ndarray
    Input image
  c : tuple as (h, w)
    The pixel willing to put in the center
    
  Returns
  -------
  A shifted image with point c at the center
  '''
  ori_c = img.shape / 2
  th, tw = ori_c[0] - c[0], ori_c[1] - c[1]
  output = np.roll(img, th, axis=0)
  output = np.roll(img, tw, axis=1)
  return output
                        
class img_show(object):
  ''' Display an image using matplotlib
  
  Args
  -------
  img
    The input image
  
  Returns
  -------
  None
  '''
  figno = 0
  def __init__(self, img):
    import matplotlib.pyplot as plt
    plt.figure(img_show.figno)
    plt.imshow(img)
    img_show.figno += 1

class img_ishow(object):
  ''' Interactive edition of img_show
  
  Allow user draw box on the image and get the box parameter. After called
  the method img_ishow.connect, one can interactivly draw a box on the image, 
  and get box parameter as a tuple (x, y, w, h) through a handler. 
  
  The method img_ishow.disconnect should be called in the handler after drawing
  
  Args
  -------
  img
    The input image
    
  handler
    A user handler function which has karg key and rect, when user draw a box
    on the image and press a key, the function will be called.
    
  Methods
  -------
  connect()
    Start to draw box
  
  disconnect()
    Stop to draw box
  '''
  def __init__(self, img):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.ion()
    self.fig = plt.subplot(111)
    self.img = self.fig.imshow(img)
    self.press = None
    self.rect = None
    self.box = self.fig.add_patch(patches.Rectangle(
                                  (0, 0), #(x, y)
                                  0,  #width
                                  0,  #height
                                  fill=False,      # remove background
                                  edgecolor="red",
                                  linewidth=3
                                              ))
    self.connect()
  def connect(self):
    self.img.figure.canvas.mpl_connect('button_press_event', self.OnPress)
    self.img.figure.canvas.mpl_connect('button_release_event', self.OnRelease)
    self.img.figure.canvas.mpl_connect('motion_notify_event', self.OnMotion)
    self.img.figure.canvas.mpl_connect('key_press_event', self.OnKeyPress)
    self.img.figure.canvas.mpl_connect('key_release_event', self.OnKeyRelease)
    #raw_input("Press Enter to continue...")
    self.img.figure.canvas.start_event_loop(timeout=-1)
  def OnPress(self, event):
    if event.inaxes!=self.img.axes: return
    #print('On Press: x=%d, y=%d, xdata=%f, ydata=%f' %
    #    (event.x, event.y, event.xdata, event.ydata))
    self.press = event.xdata, event.ydata
    self.box.set_width(0)
    self.box.set_height(0)
    self.box.figure.canvas.draw()
  def OnMotion(self, event):
    if event.inaxes!=self.img.axes: return
    #print('On Motion: x=%d, y=%d, xdata=%f, ydata=%f' %
    #    (event.x, event.y, event.xdata, event.ydata))
    if self.press == None:
      return
    self.motion = event.xdata, event.ydata
    self.x0 = min((self.press[0], self.motion[0]))
    self.y0 = min((self.press[1], self.motion[1]))
    self.w = abs(self.press[0]-self.motion[0])
    self.h = abs(self.press[1]-self.motion[1])
    self.box.set_x(self.x0)
    self.box.set_y(self.y0)
    self.box.set_width(self.w)
    self.box.set_height(self.h)
    self.box.figure.canvas.draw()
  def OnRelease(self, event):
    if event.inaxes!=self.img.axes: return
    'on release we reset the press data'
    self.press = None
    self.rect = int(self.x0), int(self.y0), int(self.x0+self.w), int(self.y0+self.h)
    self.box.figure.canvas.draw()
  def OnKeyPress(self, event):
    pass
  def OnKeyRelease(self, event):
    self.img.figure.canvas.stop_event_loop()
    self.disconnect()
    self.key = event.key
    pass
#    self.key = event.key
#    if self.handler(fig_obj = self):
#      self.fig.canvas.stop_event_loop()
  def disconnect(self):
    'disconnect all the stored connection ids'
    self.box.figure.canvas.mpl_disconnect(self.OnPress)
    self.box.figure.canvas.mpl_disconnect(self.OnRelease)
    self.box.figure.canvas.mpl_disconnect(self.OnMotion)
    self.box.figure.canvas.mpl_disconnect(self.OnKeyPress)
    self.box.figure.canvas.mpl_disconnect(self.OnKeyRelease)

if __name__ == '__main__':
## Test all functions
  # Read an image
#  fig = img_ishow(img)
#  roi = img_croproias(img, fig.rect, (64, 128), val = 128)
#  print('roi shape: %s'%(str(roi.shape)))
#  img_show(roi)
#  import matplotlib.pyplot as plt
#  x = np.arange(5)
#  y = np.exp(x)
#  plt.figure(0)
#  plt.plot(x, y)
#  
#  z = np.sin(x)
#  plt.figure(1)
#  plt.plot(x, z)
#  
#  w = np.cos(x)
#  plt.figure(0) # Here's the part I need
#  plt.plot(x, w)
#  img = img_read('/home/ziheng/git/cpm/dataset/MPI/images/033473533.jpg')
#  s = skimg.transform.rotate(img, 45, cval = 128, resize = True)
#  img_show(img)
#  s = img_rot(s, -45, center = (0, 0))
#  img_show(s)
#  s = img_flip(s, 'ud')
#  img_show(s)
#  img = img_rot(img, angle = 45, center = (0, 600))
#  img_show(img)
  img = np.random.rand(32,32)
  img_show(img)
  img = img_scale(img, f=(0.5, 0.5))
  img_show(img)
  pass