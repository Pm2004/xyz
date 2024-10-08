#!/bin/bash

# Install Packages
sudo apt update && sudo apt upgrade -y

sudo apt install ca-certificates zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev curl git wget make jq build-essential pkg-config lsb-release libssl-dev libreadline-dev libffi-dev gcc screen unzip lz4 -y

# Install Python3
sudo apt install python3 -y
python3 --version

sudo apt install python3-pip -y
pip3 --version

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io -y
docker --version

# Install Docker-Compose
VER=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)

sudo curl -L "https://github.com/docker/compose/releases/download/$VER/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version

# Docker permission to the user
sudo groupadd docker
sudo usermod -aG docker $USER

# Clone the repository
git clone https://github.com/allora-network/basic-coin-prediction-node
cd basic-coin-prediction-node || exit

# Copy the .env.example to .env
cp .env.example .env

# Copy config.example.json to config.json
cp config.example.json config.json

# Function to update .env file with user input
update_env() {
  key=$1
  value=$2
  sed -i "s/^$key=.*/$key=$value/" .env
}

# Function to update config.json with user input using jq
update_config() {
  key=$1
  value=$2
  jq --arg v "$value" ".worker[0].parameters.$key = \$v" config.json > config.tmp.json && mv config.tmp.json config.json
}

# Function to update topicId in config.json as an integer
update_topic_id() {
  key=$1
  value=$2
  jq ".worker[0].$key = $value" config.json > config.tmp.json && mv config.tmp.json config.json
}

# Prompt user for the necessary input
echo "Please select a TOKEN from the list below:"
PS3="Enter your choice (1-5): "
options=("ETH" "SOL" "BTC" "BNB" "ARB")
topic_ids=("1" "3" "5" "8" "9")
select opt in "${options[@]}"; do
  if [[ -n $opt ]]; then
    update_env "TOKEN" "$opt"
    update_config "Token" "$opt"
    update_topic_id "topicId" "${topic_ids[REPLY-1]}"
    break
  fi
done

# Prompt for TRAINING_DAYS
read -p "Enter the number of TRAINING_DAYS (Select: 2, 31): " training_days
update_env "TRAINING_DAYS" "$training_days"

# Prompt for TIMEFRAME based on the training days
echo "Please select a TIMEFRAME based on TRAINING_DAYS:"
if [[ $training_days -le 2 ]]; then
  echo "Use a TIMEFRAME of >= 30min"
elif [[ $training_days -le 30 ]]; then
  echo "Use a TIMEFRAME of >= 4h"
else
  echo "Use a TIMEFRAME of >= 4d"
fi
read -p "Enter the TIMEFRAME (Select:, 30min, 4h, 4d): " timeframe
update_env "TIMEFRAME" "$timeframe"

# Prompt for REGION
echo "Please select a REGION:"
PS3="Enter your choice (1-2): "
regions=("EU" "US")
select region in "${regions[@]}"; do
  if [[ -n $region ]]; then
    update_env "REGION" "$region"
    break
  fi
done

# Prompt for DATA_PROVIDER
echo "Please select a DATA_PROVIDER:"
PS3="Enter your choice (1-2): "
providers=("Binance" "Coingecko")
select provider in "${providers[@]}"; do
  if [[ -n $provider ]]; then
    update_env "DATA_PROVIDER" "$provider"
    break
  fi
done

# Prompt for CG_API_KEY if Coingecko is selected
if [[ $provider == "Coingecko" ]]; then
  read -p "Enter your Coingecko API Key: " cg_api_key
  update_env "CG_API_KEY" "$cg_api_key"
else
  update_env "CG_API_KEY" ""
fi

# Prompt for wallet name and seed phrase
read -p "Enter your wallet name: " wallet_name
read -p "Enter your seed phrase: " seed_phrase

# Update config.json with wallet name and seed phrase
jq --arg wallet "$wallet_name" --arg seed "$seed_phrase" \
'.wallet.addressKeyName = $wallet | .wallet.addressRestoreMnemonic = $seed' config.json > config.tmp.json && mv config.tmp.json config.json

# Create the Python script for the advanced model
cat > ml_model.py <<EOF
import torch
import torch.nn as nn
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

# Custom Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AttentionLayer, self).__init__()
        self.attn_weights = nn.Parameter(torch.Tensor(hidden_layer_size * 2, 1))
        nn.init.xavier_uniform_(self.attn_weights)

    def forward(self, lstm_output):
        attn_scores = torch.matmul(lstm_output, self.attn_weights)
        attn_scores = torch.softmax(attn_scores, dim=1)
        weighted_output = lstm_output * attn_scores
        return torch.sum(weighted_output, dim=1)

# Advanced BiLSTM Model with Attention
class AdvancedBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(AdvancedBiLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size * 2)

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size)
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        attn_out = self.attention(lstm_out)
        predictions = self.linear(attn_out)
        return predictions

# Fetch historical data from Binance
def get_binance_data(symbol="ETHUSDT", interval="1m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "close"]]
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)
        return df
    else:
        raise Exception(f"Failed to retrieve data: {response.text}")

# Prepare dataset for training
def prepare_dataset(symbols, sequence_length=10):
    all_data = []
    for symbol in symbols:
        df = get_binance_data(symbol)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
        for i in range(sequence_length, len(scaled_data) - 20):
            seq = scaled_data[i-sequence_length:i]
            label_10 = scaled_data[i+10] if i+10 < len(scaled_data) else scaled_data[-1]
            label_20 = scaled_data[i+20] if i+20 < len(scaled_data) else scaled_data[-1]
            label = torch.FloatTensor([label_10[0], label_20[0]])
            all_data.append((seq, label))
    return all_data, scaler

# Train model
def train_model(model, data, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq, label in data:
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
            optimizer.zero_grad()
            output = model(seq_tensor)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.4f}")

if __name__ == "__main__":
    input_size = 1
    hidden_layer_size = 64
    output_size = 1
    num_layers = 2
    dropout = 0.2

    model = AdvancedBiLSTMModel(input_size, hidden_layer_size, output_size, num_layers, dropout)
    symbols = ["ETHUSDT", "BTCUSDT"]
    data, scaler = prepare_dataset(symbols)
    train_model(model, data)
EOF

# Make init.config executable and run it
chmod +x init.config
./init.config

# Start Docker containers and build
docker compose up --build -d

# Output completion message
echo "Your worker node has been started. To check logs, run:"
echo "docker logs -f worker"
