# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python (image-proc)
#     language: python
#     name: image-proc
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# %%
im = cv.imread("a1images/emma.jpg")

fig, ax = plt.subplots(1,1,figsize = (3,4))
ax.imshow(cv.cvtColor(im,cv.COLOR_BGR2RGB))
ax.xaxis.set_ticks_position('top')
plt.show()


# %%
im = plt.imread("a1images/emma.jpg")
fig ,ax = plt.subplots(1,1, figsize = (3,4))

ax.imshow(im)
ax.xaxis.set_ticks_position('top')
plt.show()

# %%
im = plt.imread("a1images/emma.jpg")
fig ,ax = plt.subplots(1,4, figsize = (12,16))

ax[0].imshow(im) 
ax[1].imshow(im[...,0],cmap='Reds')    #red plane only
ax[2].imshow(im[...,1],cmap='Greens')  #green plane only
ax[3].imshow(im[...,2],cmap='Blues')   #blue plane only

#ax.xaxis.set_ticks_position('top')
plt.show()

# %%
im.shape

# %%
rows , cls , chas = im.shape   #(810, 720, 3)

gray_im = np.zeros((rows,cls))

for i in range(rows):
    for j in range(cls):
        val = sum(im[i][j])/3
        gray_im[i][j] = val

fig ,ax = plt.subplots(1,1, figsize = (3,4))

ax.imshow(gray_im)
plt.show()

# %% [markdown]
# ## Question 1

# %%
gray_im = np.dot(im,[0.2989, 0.5870, 0.1140])

fig ,ax = plt.subplots(1,1, figsize = (3,4))
ax.imshow(gray_im,cmap = 'gray')
ax.xaxis.set_ticks_position('top')
plt.show()


# %%
def f(x):
    return np.where(
        (x >= 50) & (x < 150),
        1.55 * x + 22.5,
        x
    ).astype(np.uint8)


    
new_im_1 = f(gray_im)

fig ,ax = plt.subplots(1,1, figsize = (3,4))

ax.imshow(new_im_1,cmap = 'gray')
ax.xaxis.set_ticks_position('top')
plt.show()

# %%
im = cv.imread('a1images/emma.jpg')
gray_im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)

"""
cv.imshow(" " , gray_im)
cv.waitKey(0)
cv.destroyAllWindows()
"""


# %%
#with mask and lut

r = np.arange(256)
s = np.zeros(256)

#function
slope = (255-100)/(150-50)
c = 100 - slope * 50

mask = (r>=50) & (r<150)
s[mask] = slope * r[mask] + c


m1 = r<50
s[m1] = r[m1]

m2 = r>=150
s[m2] = r[m2]

lut = np.clip(np.round(s),0,255).astype(np.uint8)

transformed = cv.LUT(gray_im,lut)


# %%
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.imshow(gray_im, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,3,2)
plt.plot(lut)
plt.title('Intensity transform (LUT)')
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')
plt.grid(True)

plt.subplot(1,3,3)
plt.imshow(transformed, cmap='gray')
plt.title('Transformed')
plt.axis('off')

plt.tight_layout()
plt.show()



# %% [markdown]
# ## Question 2

# %%
im = cv.imread("a1images/brain_proton_density_slice.png")
gray_im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)


"""
cv.imshow(" " , gray_im)
cv.waitKey(0)
cv.destroyAllWindows()
"""

# %% [markdown]
# ### part a : white matter

# %%
r = np.arange(256)
s = np.zeros_like(r)



m1 = r<120
s[m1] = 0.8 * r[m1]

m2 = (120<=r) & (r<256)
s[m2] = 1.5*r[m2] - 80


m3 = r>=230
s[m3] = 255 - 0.5 * (255 - r[m3]) 


lut = np.clip(np.round(s),0,255).astype(np.uint8)

transformed = cv.LUT(gray_im,lut)

# %%
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.imshow(gray_im, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,3,2)
plt.plot(lut)
plt.title('Intensity transform (LUT)')
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')
plt.grid(True)

plt.subplot(1,3,3)
plt.imshow(transformed, cmap='gray')
plt.title('Transformed')
plt.axis('off')

plt.tight_layout()
plt.show()



# %% [markdown]
# ### part b : gray matter 

# %%
r = np.arange(256)
s = np.zeros_like(r)


m1 = r < 70
s[m1] = 0.6 * r[m1]

m2 = (r >= 70) & (r <= 130)
s[m2] = 80 + 2 * (r[m2] - 70)  

m3 = r > 130
s[m3] = 0.7 * r[m3]

lut = np.clip(np.round(s),0,255).astype(np.uint8)
transformed = cv.LUT(gray_im,lut)

# %%
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.imshow(gray_im, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,3,2)
plt.plot(lut)
plt.title('Intensity transform (LUT)')
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')
plt.grid(True)

plt.subplot(1,3,3)
plt.imshow(transformed, cmap='gray')
plt.title('Transformed')
plt.axis('off')

plt.tight_layout()
plt.show()



# %% [markdown]
# ## Gamma Correction
#
# ---
#
# ### 1. What is Gamma?
#
# - **Gamma** controls brightness in a *non-linear* way (not just brighter or darker).
# - Human eyes see brightness **logarithmically** (we notice shadows and midtones more than extreme blacks or whites).
# - **Digital screens** store and display images in a *gamma-encoded* way so they look correct to us.
#
# ---
#
# ### 2. Why We Need Gamma Correction
#
# - Cameras capture light **linearly** (real-world brightness is proportional to light intensity).
# - If we showed this linear data directly, the image would look **too dark**.
# - **Gamma correction** brightens midtones without overexposing highlights or losing shadow detail.
#
# ---
#
# ### 3. Two Main Uses
#
# 1. **Encoding (Gamma Encoding)** — applied when saving an image, so it looks correct on screens.  
# 2. **Decoding (Gamma Decoding)** — applied when loading or processing an image, to restore real light values.
#
# ---
#
# ### 4. The Gamma Formula
#
# - **Encoding:**  
#
# $$
# \text{encoded} = \text{real}^{\frac{1}{\gamma}}
# $$
#
# - **Decoding:**  
#
# $$
# \text{real} = \text{encoded}^{\gamma}
# $$
#
# - For most displays, **γ ≈ 2.2**.
#
# ---
#
# ### 5. Practical Example
#
# - Suppose a pixel’s real light value is **0.5** (50% brightness).
# - With γ=2.2 encoding:  
#
# $$
# 0.5^{\frac{1}{2.2}} \approx 0.73
# $$
#
# - Stored pixel value looks brighter on screen.
#
# ---
#
# ### 6. Key Points to Remember
#
# - Gamma is **not** just a brightness slider — it changes *tone mapping* especially in midtones.
# - **Linear space** = good for math & image processing.  
#   **Gamma space** = good for viewing on screens.
# - Always **convert to linear** before blending, resizing, or doing lighting math in graphics.
#
# ---
#
# ### 7. Real-World Connections
#
# - **Photography** → RAW files are linear, JPEG is gamma-encoded.
# - **Video** → Uses gamma curves like Rec.709 or sRGB.
# - **Games / 3D graphics** → Lighting calculated in linear space, then gamma-encoded.
#
# ---
#
# ### 8. Mental Model
#
# Think of gamma like **stretching** and **compressing** the brightness scale:
#
# - γ > 1 → Midtones are brighter (common in displays).
# - γ < 1 → Midtones are darker.
#

# %% [markdown]
# ## Question 3

# %%
im_bgr = cv.imread("a1images/highlights_and_shadows.jpg")

im_rgb = cv.cvtColor(im_bgr,cv.COLOR_BGR2RGB)

im_lab = cv.cvtColor(im_rgb,cv.COLOR_RGB2LAB)



# %%
print(im_bgr[0])
print("*********")
print(im_rgb[0])
print("*********")
print(im_lab[0])

# %%
L, a, b = cv.split(im_lab)

# Show channels
plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(L, cmap='gray'); plt.title("L* (Lightness)")
plt.subplot(1,3,2); plt.imshow(a, cmap='gray'); plt.title("a* (Green–Red)")
plt.subplot(1,3,3); plt.imshow(b, cmap='gray'); plt.title("b* (Blue–Yellow)")
plt.show()

# %%
#gamma correction
L_f = L * (100/255)
gamma = 0.7
L_g = ((L_f/100) ** gamma) *100

L_corrected = np.clip(np.round(L_g * (255/100)),0,255).astype(np.uint8)

lab_corrected = cv.merge([L_corrected,a,b])
im_corrected = cv.cvtColor(lab_corrected,cv.COLOR_LAB2RGB)


# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# Original Image
axes[0, 0].imshow(im_rgb)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

# Corrected Image
axes[0, 1].imshow(im_corrected)
axes[0, 1].set_title(f"Gamma Corrected (γ={gamma})")
axes[0, 1].axis("off")

# L channel histogram (original)
axes[1, 0].hist(L.flatten(), bins=256, color='gray')
axes[1, 0].set_title("Original L Histogram")

# L channel histogram (corrected)
axes[1, 1].hist(L_corrected.flatten(), bins=256, color='gray')
axes[1, 1].set_title("Corrected L Histogram")

# Hide empty plot
axes[0, 2].axis("off")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()

# %%
L

# %%
L_corrected

# %% [markdown]
# ## Question 4

# %%
im = cv.imread("a1images/spider.png")
img = cv.cvtColor(im, cv.COLOR_BGR2RGB)
im_hsv = cv.cvtColor(im,cv.COLOR_BGR2HSV)

# %%
print(im[0])
print("***")
print(im_hsv[0])

# %%
H,S,V = cv.split(im_hsv)
S

# %%
a = 0.4
sigma = 70

S_cal = S + a * 128 * np.exp(-((S - 128) ** 2) / (2 * sigma ** 2))
S_en = np.clip(np.round(S_cal),0,255).astype(np.uint8)

S_en

# %%
hsv_en = cv.merge([H,S_en,V])
im_en = cv.cvtColor(hsv_en,cv.COLOR_HSV2RGB)


# %%
x_vals = np.arange(0, 256)
y_vals = np.clip(x_vals + a * 128 * np.exp(-((x_vals - 128) ** 2) / (2 * sigma ** 2)), 0, 255)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(im_en)
ax[1].set_title(f"Vibrance Enhanced (a={a})")
ax[1].axis("off")

ax[2].plot(x_vals, y_vals, color='red')
ax[2].set_title("Intensity Transformation")
ax[2].set_xlabel("Input S value")
ax[2].set_ylabel("Output S value")
ax[2].grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Histogram Equalization

# %% [markdown]
# ### Question 5  

# %%
im_gray = cv.imread("a1images/shells.tif",cv.IMREAD_GRAYSCALE)
print(im_gray.dtype.itemsize) #if uint16 then 2


# %%
def hist_eq(im):

    M,N = im.shape

    L = 2** (im.dtype.itemsize * 8)   

    temp = ((L-1) / (M*N))

    n = np.array([np.sum(im==r) for r in range(L)])

    cdf_n =  np.array([np.sum(n[:i]) for i in range(L)])

    s = temp *  cdf_n
    s = np.clip(np.round(s),0,255).astype(np.uint8)

    return s,n    


# %%
s , hist_before = hist_eq(im_gray)

im_eq = cv.LUT(im_gray,s)


hist_after = np.array([np.sum(im_eq == r) for r in range(2**(im.dtype.itemsize*8))])

print()

# %%
fig , ax = plt.subplots(1,2 , figsize= (10,5))

ax[0].plot(hist_before)
ax[0].set_title('Histogram Before')
ax[0].set_xlabel('r')
ax[0].set_ylabel('Pixel count')
ax[0].grid(True)

ax[1].plot(hist_after)
ax[1].set_title('Histogram Equalization')
ax[1].set_xlabel('Transformed ; s = T(r)')
ax[1].set_ylabel('Pixel count')
ax[1].grid(True)

# %%
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.imshow(im_gray, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,3,2)
plt.plot(s)
plt.title('Intensity transform (LUT)')
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')
plt.grid(True)

plt.subplot(1,3,3)
plt.imshow(hist_eq_trans, cmap='gray')
plt.title('Transformed')
plt.axis('off')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### vectorized version

# %%
def hist_eq(im):
    # Total number of possible intensity levels
    L = 2 ** (im.dtype.itemsize * 8)

    # Compute histogram efficiently
    hist = np.bincount(im.ravel(), minlength=L)

    # Cumulative distribution function (CDF)
    cdf = np.cumsum(hist)

    # Create mapping (lookup table)
    s = np.round((L - 1) * cdf / cdf[-1]).astype(im.dtype)

    return s, hist

# Load grayscale image
im = cv.imread("a1images/shells.tif", cv.IMREAD_GRAYSCALE)

# Get equalization mapping & original histogram
s, hist_before = hist_eq(im)

# Apply mapping
im_eq = cv.LUT(im, s)

# Histogram after equalization
hist_after = np.bincount(im_eq.ravel(), minlength=2**(im_eq.dtype.itemsize*8))

# Plot nicely
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Before
ax[0].bar(np.arange(len(hist_before)), hist_before, color='steelblue', width=1)
ax[0].set_title("Histogram Before Equalization", fontsize=14, fontweight='bold')
ax[0].set_xlabel("Pixel Value", fontsize=12)
ax[0].set_ylabel("Frequency", fontsize=12)
ax[0].set_xlim(0, len(hist_before)-1)

# After
ax[1].bar(np.arange(len(hist_after)), hist_after, color='darkorange', width=1)
ax[1].set_title("Histogram After Equalization", fontsize=14, fontweight='bold')
ax[1].set_xlabel("Pixel Value", fontsize=12)
ax[1].set_ylabel("Frequency", fontsize=12)
ax[1].set_xlim(0, len(hist_after)-1)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Question 6

# %%
im = cv.imread("a1images/jeniffer.jpg")

im_hsv = cv.cvtColor(im,cv.COLOR_BGR2HSV)

H,S,V = cv.split(im_hsv)

# %%
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.imshow(H, cmap='gray')
plt.title('H plane')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(S, cmap='gray')
plt.title('S plane')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(V, cmap='gray')
plt.title('V plane')
plt.axis('off')

plt.tight_layout()
plt.show()



# %%
_ , mask =  cv.threshold(V,120,255,cv.THRESH_BINARY)
mask

# %%
foreground = cv.bitwise_and(V, V, mask=mask)
foreground

# %%
hist = cv.calcHist([foreground], [0], mask, [256], [0, 256])
cdf = np.cumsum(hist)

# %%
plt.hist(hist.flatten(), bins=256, color='gray')
plt.show()


# %%
L = 2 ** (im.dtype.itemsize * 8)
s = np.round((L - 1) * cdf / cdf[-1]).astype(im.dtype)

foreground_eq = s[foreground]
s

# %%
foreground

# %%
hist_eq = cv.calcHist([foreground_eq], [0], mask, [256], [0, 256])
plt.hist(hist_eq.flatten(), bins=256, color='gray')
plt.show()

# %%
background_mask = cv.bitwise_not(mask)
background = cv.bitwise_and(V, V, mask=background_mask)
v_equalized = cv.add(background, foreground_eq)



# %%
hsv_equalized = cv.merge([H, S, v_equalized])
result = cv.cvtColor(hsv_equalized, cv.COLOR_HSV2BGR)

# %%
fig, ax = plt.subplots(2, 4, figsize=(15, 8))
images = [
    H, S, V, mask,
    cv.cvtColor(im, cv.COLOR_BGR2RGB),
    foreground, foreground_eq,
    cv.cvtColor(result, cv.COLOR_BGR2RGB)
]
titles = [
    "Hue", "Saturation", "Value", "Mask",
    "Original Image", "Foreground", "Equalized FG", "Final Result"
]

for i, (im, title) in enumerate(zip(images, titles)):
    ax[i // 4, i % 4].imshow(im if len(im.shape) == 2 else im)
    ax[i // 4, i % 4].set_title(title)
    ax[i // 4, i % 4].axis("off")

plt.tight_layout()
plt.show()

# %%

# %%
