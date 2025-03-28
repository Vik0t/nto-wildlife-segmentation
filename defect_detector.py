from cleanvision import Imagelab
issue_types = {"dark": {"threshold": 0.5}, 'blurry': {"threshold": 0.3}, 'light': {"threshold": 0.5}}
# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(data_path="/home/vik0t/hackaton/imgs_to_process/")

# Automatically check for a predefined list of issues within your dataset
imagelab.find_issues(issue_types)

# Produce a neat report of the issues found in your dataset
imagelab.report()

imagelab.issues()

dark_images = imagelab.issues[imagelab.issues["is_blurry_issue"] == True].sort_values(
    by=["blurry_score"]
)
dark_image_files = dark_images.index.tolist()