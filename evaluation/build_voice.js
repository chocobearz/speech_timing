const fs = require('fs');
const sdk = require("microsoft-cognitiveservices-speech-sdk");
const base_path = "../"

/*
  Arguments:
  inputFile: a json containing word length information
  template: if template use template mode
*/

function xmlToString(filePath) {
    const xml = fs.readFileSync(filePath, "utf8");
    return xml;
}

function synthesizeSpeech(ssml, audioFile) {
    const speechConfig = sdk.SpeechConfig.fromSubscription("b97af01fdfb94ac49fd2e1c86c866d4d", "canadacentral");
    const audioConfig = sdk.AudioConfig.fromAudioFileOutput(audioFile);
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig, audioConfig);
  
    //const ssml = xmlToString("neutral.xml");
    synthesizer.speakSsmlAsync(
        ssml,
        result => {
            if (result.reason == sdk.ResultReason.SynthesizingAudioCompleted) {
                console.log("synthesis finished.");
                console.log(JSON.stringify(result));
            } else {
              console.error("Speech synthesis canceled, " + result.errorDetails);
            }
  
            synthesizer.close();
            synthesizer = null;
        },
        error => {
          console.trace("err - " + err);
          synthesizer.close();
          synthesizer = null;
        });
        console.log("Now synthesizing to: " + audioFile);
}

const wordLengthFile = process.argv.slice(2);

if (wordLengthFile.length != 1 && wordLengthFile.length != 2) {
  console.log("One argument should be given: location of a json containing word length information, and optional 'template' if templating");
  process.exit();
}

let rawdata = fs.readFileSync(`${base_path}/${wordLengthFile[0]}`);
let word_lengths = JSON.parse(rawdata);

const scripts = {
    "It's eleven o'clock": "IEO",
    "That is exactly what happened" : "TIE",
    "I'm on my way to the meeting" : "IOM",
    "I wonder what this is about" : "IWW",
    "The airplane is almost full" : "TAI",
    "Maybe tomorrow it will be cold" : "MTI",
    "I would like a new alarm clock" : "IWL",
    "I think I have a doctor's appointment" : "ITH",
    "Don't forget a jacket" : "DFA",
    "I think I've seen this before" : "ITS",
    "The surface is slick" : "TSI",
    "We'll stop in a couple of minutes" : "WSI"
};

emotions = {
    "A": "angry",
    "D": "unfriendly",
    "F": "terrified",
    "H": "cheerful",
    "S": "sad",
    "N": "neutral"
}

for (key in Object.keys(word_lengths)) {
    const script = Object.keys(word_lengths)[key];
    const lengths = word_lengths[Object.keys(word_lengths)[key]];
    const script_tag = scripts[script];
    let ssml = ""

    // if you have a tamplate tag
    if (wordLengthFile.length === 2){
      if (wordLengthFile[1] === "template") {
        const synth_template = `./templates/${script_tag}_template.xml`;
        ssml = xmlToString(synth_template);

        // fill template word lengths
        lengths.forEach((length, i) => {
          length = length*100;
          ssml = ssml.replace(`RATE${i}`, `"${length}%"`);
        });
      } else {
        console.log("Incorrect template argument, only provide 'template' or leave blank");
        process.exit();
      }
    } else {
      // base with no word lengths  
      const synth_template = `./templates/azure_base/${script_tag}.xml`;
      ssml = xmlToString(synth_template);
    }
    // fill emotion tag template
    const path = wordLengthFile[0].split("/");
    const file = path.slice(-1)
    const emotion_code = emotions[file[0][0]]
    ssml = ssml.replace(`EMOTION`, `"${emotion_code}"`);

    console.log(ssml)
    const out_file = `${base_path}/audio/${emotion_code}_${script_tag}.wav`
    //generate audio
    synthesizeSpeech(ssml, out_file)
}
