class SGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        """Applying gradient descent to parameters"""
        # Implement SGD!
        
    
        var_index = 0
        for current_layer_params, current_layer_lrs in zip(parameters, learning_rate):
            for current_lr, current_param in zip(current_layer_params, current_layer_lrs):
            
                current_lr -= config['learning_rate'] * current_param
                var_index += 1 

    def zero_grad(self):
        """Resetting gradient for all parameters (set gradient to zero)"""
        for i, parameter in enumerate(self.parameters):
            self.parameters[i].grad = 0
