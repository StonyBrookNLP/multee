import torch

class EachOutputTimeDistributed(torch.nn.Module):
    """
    If module outputs list of tensors, then this module does time-distribution on each and
    return the result.
    """
    def __init__(self, module):
        super(EachOutputTimeDistributed, self).__init__()
        self._module = module

    def forward(self, *inputs):  # pylint: disable=arguments-differ
        reshaped_inputs = []
        for input_tensor in inputs:
            input_size = input_tensor.size()
            if len(input_size) <= 2:
                raise RuntimeError("No dimension to distribute: " + str(input_size))

            # Squash batch_size and time_steps into a single axis; result has shape
            # (batch_size * time_steps, input_size).
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

        reshaped_outputs_list = self._module(*reshaped_inputs)

        if not isinstance(reshaped_outputs_list, (tuple, list)):
            raise RuntimeError("Module did not return a list in EachOutputTimeDistributed")

        outputs_list = []
        for reshaped_outputs in reshaped_outputs_list:
            # Now get the output back into the right shape.
            # (batch_size, time_steps, [hidden_size])
            new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
            outputs = reshaped_outputs.contiguous().view(*new_shape)
            outputs_list.append(outputs)

        return outputs_list
