from cloudAMQP_client import CloudAMQPClient

CLOUDAMQP_URL = 'amqp://dcovrxiu:ux4FkI27XKyM64wBLtvnOxPm6t9nTZj-@sidewinder.rmq.cloudamqp.com/dcovrxiu'
QUEUE_NAME = 'dataFetcherTaskQueue'

# Initialize a client
client = CloudAMQPClient(CLOUDAMQP_URL, QUEUE_NAME)

# Send a message
client.sendDataFetcherTask({'zpid' : '83154148'})


# Receive a message
#client.getDataFetcherTask()
