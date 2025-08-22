from pydantic import BaseModel, Field
from typing import Optional

class LoanApplication(BaseModel):
    Gender: Optional[str] = Field(None, description="Male/Female")
    Married: Optional[str] = Field(None, description="Yes/No")
    Dependents: Optional[str] = Field(None, description="0/1/2/3+")
    Education: Optional[str] = Field(None, description="Graduate/Not Graduate")
    Self_Employed: Optional[str] = Field(None, description="Yes/No")
    ApplicantIncome: Optional[float] = None
    CoapplicantIncome: Optional[float] = None
    LoanAmount: Optional[float] = None
    Loan_Amount_Term: Optional[float] = None
    Credit_History: Optional[float] = None
    Property_Area: Optional[str] = Field(None, description="Urban/Semiurban/Rural")
