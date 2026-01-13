'use strict';

const express = require('express');
const path = require('path');

const app = express();
const PORT = Number(process.env.PORT) || 3000;

const WWW_DIR = path.join(__dirname, 'www');

app.disable('x-powered-by');

// 静的配信
app.use(express.static(WWW_DIR, {
  etag: true,
  fallthrough: true,
}));

// SPA フォールバック（Express 5 対応）
app.get(/.*/, (req, res) => {
  res.sendFile(path.join(WWW_DIR, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Serving ${WWW_DIR} on http://localhost:${PORT}`);
});
