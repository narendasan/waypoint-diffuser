import torch
import numpy as np
from tqdm import tqdm

from waypoint_network import WaypointMLP
from push_t_dataset import PushTStateDataset, normalize_data, unnormalize_data

class WaypointUtil():
    def __init__(self, dataset_path,
                 pred_horizon=16, batch_size=256,
                 include_agent=True):
        # parameters
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size

        # dataset and dataloader
        self.dataset_path = dataset_path
        self.dataset = PushTStateDataset(dataset_path, pred_horizon, 0, 0)
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
            self.dataset, [0.8, 0.1, 0.1])

        # print data stats
        print("start.shape:", self.dataset[0]['start'].shape)
        print("goal.shape", self.dataset[0]['goal'].shape)
        # save data stats
        self.start_dim = self.dataset[0]['start'].shape[-1]
        self.goal_dim = self.dataset[0]['goal'].shape[-1]

        # create waypoint network
        self.include_agent = include_agent
        if include_agent:
            self.waypoint_net = WaypointMLP(self.start_dim, self.goal_dim, 256, 3)
        else:
            self.waypoint_net = WaypointMLP(self.goal_dim, self.goal_dim, 256, 3)

    def to_device(self, device):
        self.device = torch.device(device)
        _ = self.waypoint_net.to(self.device)

    def train(self, num_epochs=100):
        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.waypoint_net.parameters(), lr=1e-3)

        best_loss = np.inf
        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            for _ in tglobal:
                self.waypoint_net.train()
                with tqdm(train_loader, desc='Batch', leave=False) as tepoch:
                    train_loss = []
                    for batch in tepoch:
                        start = batch['start'].to(self.device)
                        goal = batch['goal'].to(self.device)

                        # forward
                        pred = self.waypoint_net(start)
                        loss = torch.nn.functional.mse_loss(pred, goal)

                        # backward
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # logging
                        loss_cpu = loss.item()
                        train_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)

                # validation
                self.waypoint_net.eval()
                val_loss = []
                for batch in val_loader:
                    start = batch['start'].to(self.device)
                    goal = batch['goal'].to(self.device)

                    pred = self.waypoint_net(start)
                    loss = torch.nn.functional.mse_loss(pred, goal)
                    val_loss.append(loss.item())

                # logging
                train_loss = np.mean(train_loss)
                val_loss = np.mean(val_loss)
                tglobal.set_postfix(
                    train_loss=train_loss,
                    val_loss=val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    if self.include_agent:
                        torch.save(self.waypoint_net.state_dict(), './checkpoint/waypoint_net_with_agent.ckpt')
                    else:
                        torch.save(self.waypoint_net.state_dict(), './checkpoint/waypoint_net_no_agent.ckpt')

    def load_pretrain(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.waypoint_net.load_state_dict(state_dict)

    def eval(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=True)
        self.waypoint_net.eval()
        test_loss = []
        for batch in test_loader:
            start = batch['start'].to(self.device)
            goal = batch['goal'].to(self.device)

            pred = self.waypoint_net(start)
            loss = torch.nn.functional.mse_loss(pred, goal)
            test_loss.append(loss.item())

        print("test_loss:", np.mean(test_loss))

if __name__ == "__main__":
    dataset_path = 'data/pusht_cchi_v7_replay.zarr.zip'
    # waypoint_util = WaypointUtil(dataset_path, include_agent=False)
    # waypoint_util.to_device('cpu')
    # waypoint_util.train(num_epochs=500)

    waypoint_util = WaypointUtil(dataset_path, include_agent=True)
    waypoint_util.to_device('cpu')
    waypoint_util.train(num_epochs=500)
