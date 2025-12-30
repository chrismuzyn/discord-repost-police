# discord-repost-police

The purpose of this discord bot is to identify links and images that have been posted multiple times, then reply to the reposter with a generated insult.

## Future Features
- the option to delete the offending repost and dm the offendor with an explanation
- only look for reposts inside of a configurable swath of time
- hashing all attachment types, not just images

# Installation
### Prerequisites
1. This bot uses the apple csam/neuralhash that you must source yourself and place in the local directory apple-neuralhash.  THIS REPOSITORY DOES NOT DISTRIBUTE THE MODEL.  I was able to source this from an Intel Macbook.  Instructions to do so are here: https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX

You could use a different visual hash algorithm if you wish, and replace the function neuralhash in the python script.

2. You must have a postgres database set up to connect.  The settings for this database are placed in the .env file. 

### Docker instructions(Recommended):
Install docker for your platform then: 

    docker build . -t repost-police:latest
    docker run repost-police
    
### Updating the bot:
    git pull
    docker build . -t repost-police:latest
    docker run repost-police
    
    
### Outside of docker:
See the Dockerfile for instructions.

# Acknowledgements
Thanks to AsuharietYgvar for instructions on how to export the apple neuralhash algorithm: https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX

Thanks to Zach Freedman for his insult generator: https://github.com/ZackFreedman/text-generators
