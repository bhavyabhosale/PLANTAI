const mongoose = require("mongoose");

exports.connectDatabase = () => {
  mongoose
    .connect(
      "mongodb+srv://ankit:4joUOY8Y1B9En2sp@cluster0.lduulwz.mongodb.net/?retryWrites=true&w=majority"
    )
    .then((con) => console.log("Database Connected:" + con.connection.host))
    .catch((e) => console.log(e));
};
