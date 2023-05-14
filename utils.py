class RescaleToUnitRange(object):
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        unit_tensor = (tensor - min_val) / (max_val - min_val)
        return unit_tensor
    
def generate_from_data(model, data, device):
    # Move data to device
    data = data.to(device)
    # Pass data through the model
    recon_batch, _, _ = model(data)
    # Return the reconstructed data
    return recon_batch
def unnormalize(img):
    img = img * 0.2363 + 0.4823 # unnormalize
    img = img.clamp(0, 1)  # clamp the pixel values
    return img