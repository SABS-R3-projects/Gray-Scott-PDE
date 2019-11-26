import pints

class GrayScott(pints.ForwardModel):
    def n_parameters(self):
        # Return the dimension of the parameter vector. The Gray-Scott model has two
        return 2