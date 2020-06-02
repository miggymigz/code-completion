# Code Completion

This is the repository for my code completion project.

# Training Process

1. Download the repositories listed in `repository_list.txt` by running the download script as shown below:

    ```bash
    python download_dataset.py
    ```

2. Encoded the downloaded dataset inside the `repositories` folder by running the encode script as shown below:

    ```bash
    python encode.py
    ```

3. Train the model by running the following command:

    ```bash
    python train.py
    ```