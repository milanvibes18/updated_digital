# Digital Twin API Documentation

## Base URL

`http://127.0.0.1:5000`

## Authentication

All `/api/*` endpoints (except `/api/register` and `/api/login`) are protected using JWT.
You must include an `Authorization` header with a Bearer token.

`Authorization: Bearer <YOUR_ACCESS_TOKEN>`

### Auth Endpoints

#### `POST /api/register`

Registers a new user.

**Request Body:**
```json
{
    "username": "newuser",
    "password": "strongpassword123",
    "email": "user@example.com"
}