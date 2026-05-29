import torch.nn as nn
import torch
from joblib import load

class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.fc3(x)
        return x

# so that torch thinks this is a safe pickle (just found this on google)
torch.serialization.add_safe_globals([ChurnModel])

ann_model = ChurnModel(input_dim=25)
ann_model.load_state_dict(torch.load(
    "webapp/assets/models/ann_model.pth",
    map_location=torch.device('cpu') #so that it works without the gpu too
))
ann_model.eval()

rfr = load("webapp/assets/models/rfc_smote_gs.pkl")
xgc = load("webapp/assets/models/xgc_gs.joblib")




def predict_all(input_df):

    input_df = input_df.astype(float)

    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
    with torch.no_grad():
        output = ann_model(input_tensor)
    probs = torch.sigmoid(output)
    ann_pred = (probs >= 0.5).int()

    return [xgc.predict(input_df)[0], rfr.predict(input_df)[0], ann_pred.item()]