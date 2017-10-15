# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 23:24:00 2017

@author: avhadsa
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from PIL import Image

def camera_calibration():

    # prepare object points
    nx = 9 #the number of inside corners in x
    ny = 6 #the number of inside corners in y
    
    #Prepare object points like (0,0,0), (1,0,0) ... (nx,ny,0)
    objp = np.zeros((nx*ny,3),np.float32)  # 3 for 3 axes X, Y and Z   Initialize all to 0.
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)  #x and y coorinates
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
        
    # Make a list of calibration images
    images = glob.glob('C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\camera_cal\\cali*.jpg')
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    # Test undistortion on an image
    img = cv2.imread('C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\camera_cal\\calibration5.jpg')
    img_size = (img.shape[1], img.shape[0])
    print (img_size)

    # camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\output_images\\01test_undist_Chessboard.jpg',dst)
    
    # Save the camera calibration result for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\output_images\\wide_dist_pickle.p", "wb" ) )
    
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)

# Define a function that chnages the perspective
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    #undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    #gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    #ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)

    #if ret == True:
        
    # If we found corners, draw them! (just for fun)
    #cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    #offset = 100 # offset for dst points
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([[550, 500], [220, 700], [1080, 700], [800, 500]])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([[330, 350], [330, 700], [970, 700], [1090, 350]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, M, Minv

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def slidinig_window(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    #print ("OLD:", lefty.shape, leftx.shape)
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2) 

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#    plt.imshow(out_img)
#    plt.plot(left_fitx, ploty, color='yellow')
#    plt.plot(right_fitx, ploty, color='yellow')
#    plt.xlim(0, 1280)
#    plt.ylim(720, 0)

    
    if left_fit.any():
        left_line_detected=True
    else:
        left_line_detected=False
    
    if right_fit.any():
        right_line_detected=True
    else:
        right_line_detected=False    
        
    line_base_pos= ((right_fitx[0]-left_fitx[0])/2) * (3.7/700)
    
    
    return out_img, ploty, left_line_detected, line_base_pos, right_line_detected, leftx, lefty, rightx, righty, left_fit, right_fit, right_fitx, left_fitx

def skilp_slidinig_window(binary_warped,left_fit,right_fit):
    #In the next frame of video we don't need to do a blind search again
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    


    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
#    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
#    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
#    left_line_pts = np.hstack((left_line_window1, left_line_window2))
#    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
#    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
#    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
#    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
#    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
#    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    if left_fit.any():
        left_line_detected=True
    else:
        left_line_detected=False
    
    if right_fit.any():
        right_line_detected=True
    else:
        right_line_detected=False    
        
    line_base_pos= ((right_fitx[0]-left_fitx[0])/2) * (3.7/700)

    return out_img, ploty, left_line_detected, line_base_pos, right_line_detected, leftx, lefty, rightx, righty, left_fit, right_fit, right_fitx, left_fitx


def curvature(ploty,left_fit,right_fit,leftx,rightx,lefty,righty):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    #Repeat this calculation after converting  x and y values to real world space.
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    #print ("NEW:", ploty.shape, leftx.shape)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    
    lane_middle = int((rightx[10] - leftx[10])/2.)+leftx[10]

    if (lane_middle-640 > 0):
        leng = 3.66/2
        mag = ((lane_middle-640)/640.*leng)
        head = ("Right",mag)
    else:
        leng = 3.66/2.
        mag = ((lane_middle-640)/640.*leng)*-1
        head = ("Left",mag)
    
    return left_curverad, right_curverad, mag

def draw_on_original(image,warped,left_fitx,right_fitx,ploty,Minv,undist,left_curverad,line_base_posi):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    text1="Radius of curvature = " + str(left_curverad)+ " (m)"
    text2="Vehicle is " + str(line_base_posi) +" m left of center"
    cv2.putText(result,text1, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result,text2, (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    plt.imshow(result)
    return result

#create an instance of the Line() class for the left and right lane lines to keep track of recent detections and to perform sanity checks
Left_line = Line()
Right_line = Line()

def process_image(): 
    #Step1:  Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
    # prepare object points
    nx = 9 #the number of inside corners in x
    ny = 6 #the number of inside corners in y
    #camera_calibration()
    
    #Step 2: Apply a distortion correction to raw images
    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load( open( "C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\output_images\\wide_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    # Read in an image
    img = cv2.imread('C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\test_images\\test4.jpg')
    img_size = (img.shape[1], img.shape[0])
    
    img_dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\output_images\\02test_undist_test4.jpg',img_dst)
    
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(img_dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    
    #Step3: Use color transforms, gradients, etc., to create a thresholded binary image
    
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img_dst, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(img_dst, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img_dst, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img_dst, sobel_kernel=ksize, thresh=(0.7, 1.3))
    hls_binary = hls_select(img_dst, thresh=(100, 255))
    
    # Combine the thresholding functions
    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
    combined[((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
             
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_dst)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    #combined1 = Image.fromarray(combined, 'RGB')
    #combined1 = combined.convertTo(combined,cv2.CV_8UC3,255.0)
    #combined1 = combined*255.0 
    #combined1= combined1.astype(np.float32)
    #combined1 = cv2.cvtColor(combined1, cv2.COLOR_GRAY2RGB)
    #print("Combined", combined1)
    cv2.imwrite('C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\output_images\\03thresholded_binary_test4.jpg',combined*255.0)
    
    #Step4: Apply a perspective transform to rectify binary image ("birds-eye view")
    warped_img, perspective_M, Minv = corners_unwarp(combined, nx, ny, mtx, dist)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(combined, cmap='gray')
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped_img, cmap='gray')
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    cv2.imwrite('C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\output_images\\04Perspective_transform_test4.jpg',warped_img*255.0)
    
    #Step5: Detect lane pixels and fit to find the lane boundary
    
    if Left_line.detected == False:
        lane_line_img, ploty, left_line_detected, line_base_posi, right_line_detected, leftx, lefty, rightx, righty, left_fit, right_fit, right_fitx, left_fitx = slidinig_window(warped_img)
    else:
        lane_line_img, ploty, left_line_detected, line_base_posi, right_line_detected, leftx, lefty, rightx, righty, left_fit, right_fit, right_fitx, left_fitx = skilp_slidinig_window(warped_img,Left_line.current_fit,Right_line.current_fit)
    
    cv2.imwrite('C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\output_images\\05lane_line_img_test4.jpg',lane_line_img)    
    #Update the Line object
    Left_line.detected = left_line_detected  
    Left_line.recent_xfitted.append(leftx) 
    Left_line.current_fit =left_fit 
    Left_line.line_base_pos = line_base_posi 
    Left_line.allx = leftx  
    Left_line.ally = lefty
    
    print(left_line_detected,right_line_detected, left_fit,right_fit,line_base_posi)
    
    Right_line.detected = right_line_detected  
    Right_line.recent_xfitted.append(rightx)
    Right_line.current_fit = right_fit   
    Right_line.line_base_pos = line_base_posi 
    Right_line.allx = rightx  
    Right_line.ally = righty
    
    
    #Step6:Determine the curvature of the lane and vehicle position with respect to center
    left_curverad, right_curverad, mag = curvature(ploty,left_fit,right_fit,leftx,rightx,lefty,righty)
    
    #Step7: Warp the detected lane boundaries back onto the original image.
    #Step8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
    result = draw_on_original(img,warped_img,left_fitx,right_fitx,ploty,Minv,img,left_curverad,mag)
    
    #print("Result", result)
    cv2.imwrite('C:\\Users\\avhadsa\\Documents\\GitHub\\CarND-Advanced-Lane-Lines\\output_images\\07result_test4.jpg',result)
    return result

process_image()

#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#
#white_output = 'output_video.mp4'
#clip1 = VideoFileClip("project_video.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)

