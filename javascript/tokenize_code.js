var esprima = require("esprima");
var fs = require("fs");
var readline = require('readline');

var jsFileList = process.argv[2]
var jsFilesHome = process.argv[3]
var tokenFile = process.argv[4]


function processFile(inputFile) {
    var instream = fs.createReadStream(inputFile),
    outstream = new (require('stream'))(),
    rl = readline.createInterface(instream, outstream);

    var writer = fs.createWriteStream(tokenFile, {
        flags: 'a' // 'a' means appending (old data will be preserved)
    })

    rl.on('line', function (line) {
        // lines.push(line);
        // console.log(line);
        var js = fs.readFileSync(jsFilesHome + line, {encoding: "utf8"});
        var tokens = esprima.tokenize(js);
        // console.log(tokens);
        for(var i = 0; i < tokens.length; i++) {
            var token = tokens[i]["value"];
            token.replace(' ', '<SP>')
            token.replace('\n', '\\n')
            token.replace('\r', '\\r')
            token.replace('\t', '\\t')
            writer.write(token);
            writer.write(" ");
        }
        // console.log("\n");
        writer.write("\n");
    });
    
    rl.on('close', function (line) {
        
    });
}

// lines = 
processFile(jsFileList);

// for (var line in lines){
    
//     break;
//     // writer.write("\n");
// }

// 
// var tokens = esprima.tokenize(js);
// fs.writeFileSync(tokenFile, JSON.stringify(tokens, 0, 2));