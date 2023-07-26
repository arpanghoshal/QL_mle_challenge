# QL_mle_challenge

# README

## How to Use

Before you begin, make sure you have all the necessary dependencies installed in your Python environment. You can install them using the command `pip install -r requirements.txt`.

The application is a FastAPI web server that you can run on your local machine. It offers endpoints for training a Gaussian Mixture Model (GMM) and retrieving the parameters of the trained model.

To run the application:

1. **Start the server**: Go to the directory that contains `main.py` and type `uvicorn main:app --reload` in your terminal. The `--reload` flag enables hot reloading, which automatically updates the server when you make changes to the code.

2. **Interact with the endpoints**: After the server is up and running, you can interact with it using HTTP requests. Key endpoints include:
    - `GET /`: Retrieves the main page of the web application.
    - `POST /update`: Updates the model with new data points, retrains it, and returns the updated parameters. This endpoint requires a `Points` object and `n_components` and `epochs` parameters in the request body.
    - `POST /train`: Trains the model with data points and returns the parameters. This endpoint requires a `Points` object and `n_components` and `epochs` parameters in the request body.
    - `POST /clear`: Clears the model and optimizer.
3. **Interact with the webpage**: http://0.0.0.0/

## Implementation Details

The application I built implements a machine learning pipeline using Gaussian Mixture Models (GMM), which allows for training and prediction based on provided data. The application uses FastAPI, PyTorch, and a number of custom modules.

Key components of the application include:

1. **Model class**: This is a custom class that encapsulates the GMM model, its optimizer, and data handling mechanisms. The class contains methods for initializing the model, updating data, and training.

2. **GMMSimple class**: This is a custom GMM implementation built using PyTorch. It has methods for forward propagation, log-probability calculation, and sample generation.

3. **FastAPI endpoints**: These are HTTP endpoints that serve as the interface for the application. The key endpoints are "/", which serves the HTML content; "/update", which allows for model training with new data; "/train", for model training; and "/clear", which clears the model.

4. **Helper functions**: These include `adjust_epochs_based_on_points`, `train_model`, and `extract_gmm_parameters`, which assist with the dynamic adjustment of training epochs, model training, and parameter extraction, respectively.

The overarching idea was to create a web service that lets users train GMM models incrementally, adjust training epochs based on the number of data points, and extract model parameters.

## Challenges Faced

The main challenge I faced in building this application was handling the incremental learning approach. Normally, for implementing online learning, one might look at libraries like Creme for inspiration. However, Creme doesn't provide an online learning implementation for GMMs.

After some research, I found that the Expectation-Maximization (EM) method, which is typically used with GMMs, could be adapted for an incremental approach. This was gleaned from resources such as [This Tutorial](https://python-course.eu/machine-learning/expectation-maximization-and-gaussian-mixture-models-gmm.php).

## Future Improvements

Given more time, my goal would be to implement a full incremental GMM using the EM algorithm. The steps would be:

`
Initialize GMM parameters: mixture_coefficients, means, covariances, N (counts per component)
For each new data point x:
    
    # E-step
    Calculate responsibilities for x given current parameters:
        For each component i:
            responsibility[i] = mixture_coefficients[i] * Gaussian(x, means[i], covariances[i])
        Normalize responsibilities so they sum to 1
    
    # Incremental M-step
    For each component i:
        # Update counts
        N_previous = N[i]
        N[i] = N[i] + responsibility[i]
        
        # Update means
        means_previous = means[i]
        means[i] = means[i] + responsibility[i] * (x - means[i]) / N[i]
        
        # Update covariances
        covariances[i] = (N_previous/N[i]) * covariances[i] + responsibility[i] * np.outer(x - means_previous, x - means[i])
        
        # Update mixture coefficients
        mixture_coefficients[i] = N[i] / total_data_points
`

This approach would make the model more flexible and efficient, particularly for streaming data. Other potential improvements include better data validation mechanisms and optimization of the model training process.

## Testing Strategy

I recommend testing the application using a combination of unit and integration tests.

**Unit Tests**: 
These tests are written to check individual functions, verifying that they perform correctly in isolation. For example, Python's unittest library could be used to confirm if the `init_model` function in the Model class correctly initializes the model and optimizer. Similarly, methods like `train_and_extract` and `update_data` could be tested to ensure they work as expected.

**Integration Tests**: 
These tests are designed to confirm that different parts of the system work together as expected. FastAPI's TestClient class could be used to test the API endpoints. For instance, the `/update` and `/clear` endpoints could be tested by checking response status codes and verifying expected keys are in the response JSON.
