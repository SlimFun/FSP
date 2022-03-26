from torch import nn
import utils

import wandb

class Server:
    def __init__(self, model, device) -> None:
        self.device = device
        self.model = model(self.device)
        self.masks = None

        # torch.save(self.model.state_dict(), 'ori_init_model.pt')

    def get_global_params(self):
        return self.model.cpu().state_dict()

    def set_global_params(self, params_dict):
        self.model.load_state_dict(params_dict)

    def _merge_local_masks(self, keep_masks_dict):
    #keep_masks_dict[clients][params]
        print('merge local masks')
        for m in range(len(keep_masks_dict[0])):
            for client_id in keep_masks_dict.keys():
            # for j in range(0, len(keep_masks_dict.keys())):
                if client_id != 0:
                    keep_masks_dict[0][m] += keep_masks_dict[client_id][m]
        # for i in range(len(keep_masks_dict[0])):
        #     for j in range(1, len(keep_masks_dict.keys())):
        #         # params = self.keep_masks_dict[0][j].to('cpu')
        #         keep_masks_dict[0][i] += keep_masks_dict[j][i]
        #         if j == len(self.keep_masks_dict.keys())-1:
        #             diff = self.keep_masks_dict[0][i].view(-1).cpu().numpy()
        #             self.keep_masks_dict[0][i] = np.where(diff, 0, 1)
        return keep_masks_dict[0]

    def _prune_global_model(self, masks):
        prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
                layer, nn.Linear), self.model.modules())

        for layer, keep_mask in zip(prunable_layers, masks):
            assert (layer.weight.shape == keep_mask.shape)

            layer.weight.data[keep_mask == 0.] = 0.

    def aggregate(self, keep_masks_dict, model_list, round):
        last_params = self.get_global_params()

        training_num = sum(local_train_num for (local_train_num, _) in model_list)

        (num0, averaged_params) = model_list[0]
        if (averaged_params is not None):
            for k in averaged_params.keys():
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w

            # for name, param in averaged_params.items():
            for name in last_params:
                assert (last_params[name].shape == averaged_params[name].shape)
                averaged_params[name] = averaged_params[name].type_as(last_params[name])
                # last_params[name] = last_params[name].type_as(averaged_params[name])
                last_params[name] += averaged_params[name]
            self.set_global_params(last_params)

        if (keep_masks_dict[0] is not None):
            if self.masks is None:
                self.masks = self._merge_local_masks(keep_masks_dict)

            # applyed_masks = copy.deepcopy(self.masks)
            # for i in range(len(applyed_masks)):
            #     applyed_masks[i] = applyed_masks[i].cpu().numpy().tolist()
            # with open(f'./applyed_masks.txt', 'a+') as f:
            #     f.write(json.dumps(applyed_masks))
            #     f.write('\n')

            self._prune_global_model(self.masks)
        

    def test_global_model_on_all_client(self, clients, round):
        # pruned_c = 0.0
        # total = 0.0
        # for name, param in self.model.state_dict().items():
        #     a = param.view(-1).to(device='cpu', copy=True).numpy()
        #     pruned_c +=sum(np.where(a, 0, 1))
        #     total += param.numel()
        # print(f'global model zero params: {pruned_c / total}')

        train_accuracies, train_losses, test_accuracies, test_losses = utils.evaluate_local(clients, self.model, progress=True,
                                                    n_batches=0)
        wandb.log({"Train/Acc": sum(train_accuracies.values())/len(train_accuracies.values())}, step=round)
        wandb.log({"Train/Loss": sum(train_losses.values())/len(train_losses.values())}, step=round)
        print(f'round: {round}')
        print(f'Train/Acc : {sum(train_accuracies.values())/len(train_accuracies.values())}; Train/Loss: {sum(train_losses.values())/len(train_losses.values())};')

        wandb.log({"Test/Acc": sum(test_accuracies.values())/len(test_accuracies.values())}, step=round)
        wandb.log({"Test/Loss": sum(test_losses.values())/len(test_losses.values())}, step=round)
        print(f'Test/Acc : {sum(test_accuracies.values())/len(test_accuracies.values())}; Test/Loss: {sum(test_losses.values())/len(test_losses.values())};')

