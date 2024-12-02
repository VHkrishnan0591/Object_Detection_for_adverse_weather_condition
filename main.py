from Image_preprocessing import *
import matplotlib.pyplot as plt
import os

def denoising_of_image(folder_path,directory_path,denoiser):
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for i in file_names:
        img = cv2.imread(os.path.join(folder_path, i))      
        if denoiser == "Rain":
            img_derained = ImageDerained()
            final_image = img_derained.derainer(img)
            final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        elif denoiser == "Fog":
            img_dehazed = ImageDehazer()
            HazeCorrectedImg = img_dehazed.dehazer(img)
            final_image = img_dehazed.increase_brightness(HazeCorrectedImg,value=60)
            final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            # final_image = img_dehazed.image_enhancer(HazeCorrectedImg)
        
        if not os.path.exists(directory_path):
            # Create the directory
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
            cv2.imwrite(os.path.join(directory_path, i), final_image)
        else:
            print(f"Directory '{directory_path}' already exists.")
            cv2.imwrite(os.path.join(directory_path, i), final_image)



directory_path_of_fog = "Processed_Image/dehazed_images"
# directory_path_of_fog = "Test _Folder\Processed\Fog"
folder_path_of_fog = "766ygrbt8y-3\DAWN\Fog\Fog"
# folder_path_of_fog = "Test _Folder\Raw\Fog"
denoising_of_image(folder_path_of_fog,directory_path_of_fog, "Fog")
print("The dehazing is completed successfully")
folder_path_of_rain = "766ygrbt8y-3\DAWN\Rain\Rain"
directory_path_of_rain = "Processed_Image/derained_images_v2_drdh"
# directory_path_of_rain = "Test _Folder\Processed\Rain"
# folder_path_of_rain = "Test _Folder\Raw\Rain"
denoising_of_image(folder_path_of_rain,directory_path_of_rain, "Rain")
print("The deraining is completed successfully")

derained_filenames = [f for f in os.listdir(directory_path_of_rain) if os.path.isfile(os.path.join(directory_path_of_rain, f))]
print(f"The length of derained file is {len(derained_filenames)}")
dehazed_filenames = [f for f in os.listdir(directory_path_of_fog) if os.path.isfile(os.path.join(directory_path_of_fog, f))]
print(f"The length of derained file is {len(dehazed_filenames)}")











# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
# axs[1].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
# axs[0].set_title('input image')
# axs[1].set_title('Weighted Corrected image')
# plt.show()

